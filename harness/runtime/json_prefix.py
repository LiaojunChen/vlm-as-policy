"""Incremental validator for the bounded JSON Schema subset used by the policy.

It accepts only prefixes which obey structural/key/enum rules. Numeric ranges
are checked at value boundaries. Semantic robot preconditions remain separate.
"""
import json
import math
import re


class Invalid(ValueError):pass
class Incomplete(Exception):pass


class JsonPrefix:
    def __init__(self,schema):self.schema=schema

    def status(self,text):
        self.text=text
        try:
            end=self.value(0,self.schema)
            return 'complete' if self.space(end)==len(text) else 'invalid'
        except Incomplete:return 'prefix'
        except (Invalid,ValueError,TypeError,OverflowError):return 'invalid'

    def space(self,index):
        while index<len(self.text) and self.text[index] in ' \t\r\n':index+=1
        return index

    def char(self,index):
        if index>=len(self.text):raise Incomplete()
        return self.text[index]

    def string(self,index,allowed=None):
        if self.char(index)!='"':raise Invalid()
        start=index;index+=1;decoded=''
        while True:
            char=self.char(index)
            if char=='"':
                value=json.loads(self.text[start:index+1])
                if allowed is not None and value not in allowed:raise Invalid()
                return value,index+1
            if char=='\\':
                escape=self.char(index+1)
                if escape=='u':
                    digits=''
                    for offset in range(2,6):
                        digit=self.char(index+offset)
                        if digit not in '0123456789abcdefABCDEF':raise Invalid()
                        digits+=digit
                    decoded+=chr(int(digits,16));index+=6
                elif escape in '"\\/bfnrt':
                    decoded+=json.loads('"\\'+escape+'"');index+=2
                else:raise Invalid()
            elif ord(char)<32:raise Invalid()
            else:decoded+=char;index+=1
            if allowed is not None and not any(word.startswith(decoded) for word in allowed):raise Invalid()

    def value(self,index,schema):
        index=self.space(index)
        if 'anyOf' in schema:
            partial=False
            for variant in schema['anyOf']:
                try:return self.value(index,variant)
                except Incomplete:partial=True
                except Invalid:pass
            if partial:raise Incomplete()
            raise Invalid()
        kind=schema['type'];char=self.char(index)
        if kind=='object':
            if char!='{':raise Invalid()
            props=schema.get('properties',{});required=set(schema.get('required',[]));seen=set()
            index=self.space(index+1)
            if self.char(index)=='}':
                if required:raise Invalid()
                return index+1
            while True:
                key,index=self.string(index,set(props)-seen)
                seen.add(key);index=self.space(index)
                if self.char(index)!=':':raise Invalid()
                index=self.space(self.value(index+1,props[key]))
                char=self.char(index)
                if char=='}':
                    if not required<=seen:raise Invalid()
                    return index+1
                if char!=',' or seen==set(props):raise Invalid()
                index=self.space(index+1)
        if kind=='array':
            if char!='[':raise Invalid()
            index=self.space(index+1);count=0
            if self.char(index)==']':
                if schema.get('minItems',0):raise Invalid()
                return index+1
            while True:
                if count>=schema.get('maxItems',100):raise Invalid()
                index=self.space(self.value(index,schema['items']));count+=1
                char=self.char(index)
                if char==']':
                    if count<schema.get('minItems',0):raise Invalid()
                    return index+1
                if char!=',' or count>=schema.get('maxItems',100):raise Invalid()
                index=self.space(index+1)
        if kind=='string':
            value,end=self.string(index,schema.get('enum'))
            if not schema.get('minLength',0)<=len(value)<=schema.get('maxLength',10000):raise Invalid()
            return end
        if kind in ('boolean','null'):
            choices=['null'] if kind=='null' else ['true','false']
            if 'enum' in schema:choices=[json.dumps(value) for value in schema['enum']]
            remaining=self.text[index:]
            for word in choices:
                if remaining.startswith(word):return index+len(word)
            if any(word.startswith(remaining) for word in choices):raise Incomplete()
            raise Invalid()
        if kind not in ('number','integer'):raise Invalid('Unsupported schema type')
        end=index
        while end<len(self.text) and self.text[end] not in ',}] \t\r\n':end+=1
        number=self.text[index:end]
        if end==len(self.text):
            # A number may still acquire a fractional/exponent suffix.
            if number.startswith('-') and schema.get('minimum',-math.inf)>=0:raise Invalid()
            if number=='-' or re.fullmatch(r'-?(?:0|[1-9][0-9]*)(?:\.[0-9]*)?',number) or re.fullmatch(r'-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?[eE][+-]?[0-9]*',number):raise Incomplete()
            raise Invalid()
        if not re.fullmatch(r'-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?',number):raise Invalid()
        value=float(number)
        if not math.isfinite(value):raise Invalid()
        if kind=='integer' and not value.is_integer():raise Invalid()
        if not schema.get('minimum',-math.inf)<=value<=schema.get('maximum',math.inf):raise Invalid()
        if 'enum' in schema and value not in schema['enum']:raise Invalid()
        return end


class GreedyJsonConstraint:
    """Choose the highest-logit token whose decoded JSON prefix is admissible."""
    def __init__(self,tokenizer,schema,prompt_length,eos_ids):
        self.tokenizer=tokenizer;self.validator=JsonPrefix(schema);self.prompt_length=prompt_length
        self.eos=set(eos_ids);self.special=set(tokenizer.all_special_ids)
        self.checked_tokens=0

    def __call__(self,input_ids,scores):
        import torch
        for batch in range(scores.shape[0]):
            generated=input_ids[batch,self.prompt_length:].tolist()
            current=self.tokenizer.decode(generated,skip_special_tokens=True,clean_up_tokenization_spaces=False)
            checked=set();selected=None
            for count in (64,512,4096,scores.shape[-1]):
                ids=torch.topk(scores[batch],min(count,scores.shape[-1])).indices.tolist()
                for token in ids:
                    if token in checked:continue
                    checked.add(token);self.checked_tokens+=1
                    if token in self.eos:
                        valid=self.validator.status(current)=='complete'
                    elif token in self.special:valid=False
                    else:
                        text=self.tokenizer.decode(generated+[token],skip_special_tokens=True,clean_up_tokenization_spaces=False)
                        valid=text!=current and self.validator.status(text)!='invalid'
                    if valid:selected=token;break
                if selected is not None:break
            if selected is None:raise ValueError('No valid JSON continuation; refusing unconstrained fallback')
            value=scores[batch,selected].clone();scores[batch].fill_(-float('inf'));scores[batch,selected]=value
        return scores
