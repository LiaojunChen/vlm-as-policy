"""Audited hosted-VLM/Codex transports for the unmodified Show-Harness roles."""
from __future__ import annotations
import base64, json, os, subprocess, tempfile, time
from pathlib import Path
from showharness.runtime import ensure_upstream
ROOT = Path(__file__).resolve().parent
CODEX_MODEL = 'gpt-5.6-luna'
ensure_upstream()
from core.vlm.vlm_client import VLMClient

class AuditedClient(VLMClient):
    def __init__(self, model, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.calls = 0
        self.last_error = None
        # The local PhysBrain endpoint supports schema-constrained decoding.
        # Hosted MaaS endpoints continue to use the historically compatible
        # json_object mode below.
        self.structured_output = model == 'physbrain1.5-8b-local' or os.environ.get('VLM_JSON_SCHEMA')=='1'
        self.codex = model == 'codex'
        super().__init__(os.environ.get('VLM_ENDPOINT', ''), model,
                         os.environ.get('VLM_API_KEY', ''), 90, 1024, 0,
                         provider='openai', max_retries=6,
                         retry_base_delay_s=2.0,retry_max_delay_s=60.0)
        if self.base_url.startswith(('http://127.0.0.1:', 'http://localhost:')):
            self.session.trust_env = False

    def complete_text(self, *args, **kwargs):
        """Use an object-only schema for PhysBrain's JSON action prompts.

        RoboDawn's planner and visual grounder both request JSON objects, but
        they have different property sets.  A permissive object schema keeps
        their existing prompts and validators authoritative while preventing
        the model's otherwise common top-level array response.
        """
        schema=kwargs.pop('response_schema',None)
        if not self.structured_output or (schema is None and self.model!='physbrain1.5-8b-local'):
            return super().complete_text(*args, **kwargs)
        kwargs.pop('strip_reasoning', None)
        try:
            return super().complete_json(*args, schema=schema or {'type': 'object'}, **kwargs)
        except RuntimeError as exc:
            # Only this local decoder rejection means invalid model output.
            # Authentication, transport and other provider errors remain fatal.
            prefix = 'VLM chat completion failed: HTTP 400: '
            body = None
            if str(exc).startswith(prefix):
                try:
                    body = json.loads(str(exc)[len(prefix):])
                except (ValueError, TypeError):
                    pass
            if (isinstance(body, dict) and isinstance(body.get('error'), dict)
                    and body['error'].get('message') == 'Generation budget ended before completing schema-constrained JSON'):
                from robotwin_harness_v3 import SkillError
                raise SkillError('Model generation ended before completing JSON; return a shorter valid response.') from exc
            raise

    def _finalize_payload(self, payload):
        # MaaS uses standard Chat Completions max_tokens; no vLLM-only fields.
        out = {k:payload[k] for k in ('model','messages','temperature','max_tokens') if k in payload}
        if 'guided_json' in payload or 'response_format' in payload:
            out['response_format'] = {'type':'json_object'}
            if self.structured_output:
                if 'guided_json' in payload:
                    out['response_format']={'type':'json_schema','json_schema':{
                        'name':'robot_decision','strict':True,'schema':payload['guided_json']}}
                else:
                    out['response_format']=payload['response_format']
        return out

    def _post_chat(self, payload):
        index = self.calls
        self.calls += 1
        prefix = self.directory / f'call_{index:04d}'
        # Store exactly what the model sees, replacing base64 with separate images.
        audit = json.loads(json.dumps(self._finalize_payload(payload)))
        images, texts = [], []
        for message in audit['messages']:
            content = message['content']
            if isinstance(content, str):
                texts.append(content)
                continue
            for part in content:
                if part.get('type') == 'text':
                    texts.append(part['text'])
                elif part.get('type') == 'image_url':
                    url = part['image_url']['url']
                    if not url.startswith('data:image/'):
                        raise ValueError('Only local observation images are allowed')
                    file = prefix.with_name(prefix.name + f'_image{len(images)}.jpg')
                    file.write_bytes(base64.b64decode(url.split(',', 1)[1]))
                    images.append(file)
                    part['image_url']['url'] = file.name
        prefix.with_suffix('.request.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
        started = time.monotonic()
        try:
            if self.codex:
                # A fresh temporary cwd prevents access to task source/expert solutions.
                with tempfile.TemporaryDirectory(prefix='robotwin-codex-') as cwd:
                    output = Path(cwd) / 'answer.txt'
                    cmd = ['codex','exec','--ephemeral','--skip-git-repo-check',
                           '--model',CODEX_MODEL,
                           '--sandbox','read-only','--json','--output-last-message',str(output),
                           '-c','model_reasoning_effort="low"','-C',cwd]
                    for file in images:
                        cmd.extend(['--image',str(file)])
                    cmd.append('-')
                    env = os.environ.copy()
                    env.pop('VLM_API_KEY', None)
                    prompt = ('You are only a visual robot policy. Use the attached images and prompt. '
                              'Do not call tools, inspect files, execute commands, access networks, or ask questions. '
                              'Return only the requested JSON or action answer.\n\n'+'\n'.join(texts))
                    p = subprocess.run(cmd,input=prompt,text=True,capture_output=True,env=env,timeout=240, encoding="utf-8")
                    prefix.with_suffix('.codex.jsonl').write_text(p.stdout, encoding="utf-8")
                    if p.returncode:
                        raise RuntimeError('Codex failed: '+p.stderr[-1500:])
                    events = [json.loads(line) for line in p.stdout.splitlines() if line.startswith('{')]
                    forbidden = {'command_execution','mcp_tool_call','web_search'}
                    if any(e.get('item',{}).get('type') in forbidden for e in events):
                        raise RuntimeError('Codex used a tool; episode excluded from visual-only evaluation')
                    raw = output.read_text(encoding="utf-8").strip()
                    usage = next((e.get('usage') for e in reversed(events) if e.get('type')=='turn.completed'),{})
                    data = {'choices':[{'message':{'content':raw}}], 'usage':usage, 'model':CODEX_MODEL}
                    elapsed = time.monotonic()-started
            else:
                data = super()._post_chat(payload)
                elapsed = time.monotonic()-started
            prefix.with_suffix('.response.json').write_text(json.dumps({'elapsed_s':elapsed,'response':data},ensure_ascii=False,indent=2), encoding="utf-8")
            self.last_error = None
            return data
        except Exception as exc:
            self.last_error = str(exc)
            prefix.with_suffix('.error.json').write_text(json.dumps({'error':str(exc),'elapsed_s':time.monotonic()-started}), encoding="utf-8")
            raise
