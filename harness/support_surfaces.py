"""Current RGB-D evidence for named support surfaces, independent of labels."""
import cv2
import numpy as np
from container_geometry import container_support


def observed_support_surface(points, table_height, held_span=()):
    points=np.asarray(points,float)
    points=points[np.isfinite(points).all(1)]
    points=points[(points[:,2]>=table_height-.005)&(points[:,2]<table_height+.38)]
    raised=points[points[:,2]>table_height+.004]
    if len(raised)<60 or np.max(np.ptp(raised[:,:2],axis=0))>.4:return None
    # A floor is a cavity only with actually observed surrounding elevated
    # walls. A convex hull or a table plane under a handle is not enclosure.
    floor=container_support(points,table_height)
    if floor is not None and floor['top']>table_height+.004:
        above=raised[raised[:,2]>floor['top']+.018]
        radial=above[:,:2]-np.asarray(floor['tcp'][:2])
        radial=radial[np.linalg.norm(radial,axis=1)>floor['free_radius_m']*.8]
        counts=np.histogram(np.arctan2(radial[:,1],radial[:,0]),
                            bins=np.linspace(-np.pi,np.pi,13))[0]
        sectors=int(np.sum(counts>=4))
        if sectors>=8:
            floor['support_inference']=dict(kind='observed_enclosed_floor',wall_sectors=sectors,
                                            required_wall_sectors=8)
            return floor
    # Otherwise require a broad, nearly horizontal upper plateau. Thin rims,
    # handles, narrow rails, curved tops and isolated depth outliers abstain.
    height=float(np.quantile(raised[:,2],.9))
    upper=raised[np.abs(raised[:,2]-height)<.003]
    if len(upper)<60 or len(upper)<len(raised)*.15:return None
    rectangle=cv2.minAreaRect(upper[:,:2].astype(np.float32))
    sizes=np.sort(np.asarray(rectangle[1],float))
    if sizes[0]<.025 or sizes[1]>.3:return None
    footprint=np.asarray(held_span,float)
    if footprint.shape==(2,) and np.isfinite(footprint).all():
        if np.any(np.sort(footprint)+.004>sizes):return None
    # The top must occupy its bounding rectangle, not be disconnected rails
    # enclosing an unobserved interior. Check observed coverage on a 4 mm grid.
    centre=np.asarray(rectangle[0]);theta=np.deg2rad(rectangle[2])
    rotation=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    local=(upper[:,:2]-centre)@rotation
    cells=np.floor(local/.004).astype(int)
    area=max(float(np.prod(rectangle[1])),1e-9)
    coverage=len(np.unique(cells,axis=0))*.004**2/area
    central=upper[np.linalg.norm(upper[:,:2]-centre,axis=1)<sizes[0]*.3]
    if coverage<.55 or len(central)<12:return None
    z=float(np.median(central[:,2]))
    if np.quantile(np.abs(central[:,2]-z),.95)>.003:return None
    return dict(tcp=[*centre.tolist(),z],top=z,source='visible_rgbd_solid_support',pixels=len(upper),
                support_inference=dict(kind='observed_upper_plateau',coverage=float(coverage),
                                       surface_size_m=sizes.tolist()))
