import numpy as np, cv2
from PIL import Image
SRC='/mnt/user-data/uploads/a_retro_cyberpunk_90s_crt_television_monitor_scen.png'
orig=np.asarray(Image.open(SRC).convert('RGB')).astype(np.float32)
H,W=orig.shape[:2]
rng=np.random.default_rng(7)

def inpaint_rows(img,x0,x1,y0,y1,lx,rx,feather=7):
    out=img.copy()
    t=np.linspace(0,1,x1-x0)[None,:,None]
    fill=np.zeros((y1-y0,x1-x0,3),np.float32)
    for y in range(y0,y1):
        L=img[y,lx[0]:lx[1]].mean(0); R=img[y,rx[0]:rx[1]].mean(0)
        fill[y-y0]=(L[None,None,:]*(1-t)+R[None,None,:]*t)[0]
    fill+=rng.normal(0,1.6,(y1-y0,x1-x0,1))
    a=np.ones((y1-y0,x1-x0),np.float32)
    f=feather
    ramp=lambda n:np.minimum(1,(np.arange(n)+1)/f)
    a*=ramp(x1-x0)[None,:]*ramp(x1-x0)[::-1][None,:]
    a*=ramp(y1-y0)[:,None]*ramp(y1-y0)[::-1][:,None]
    a=a[...,None]
    out[y0:y1,x0:x1]=img[y0:y1,x0:x1]*(1-a)+fill*a
    return out

def make_clean():
    im=orig.copy()
    im=inpaint_rows(im,216,420,212,275,(426,436),(426,436),feather=10)      # CH 01
    im=inpaint_rows(im,880,1012,205,275,(866,878),(1014,1024))   # PLAY
    im=inpaint_rows(im,240,1012,288,456,(222,236),(1014,1024))   # title+subtitle
    im=inpaint_rows(im,752,806,766,806,(742,750),(808,818))      # cursor
    return np.clip(im,0,255)

# glass mask (superellipse)
def glass_mask(shrink=0):
    yy,xx=np.mgrid[0:H,0:W].astype(np.float32)
    cx,cy,a,b,n=618,503,447-shrink,328-shrink,4.8
    m=(np.abs((xx-cx)/a)**n+np.abs((yy-cy)/b)**n)<=1
    m=cv2.GaussianBlur(m.astype(np.float32),(0,0),2.0)
    return m
if __name__=='__main__':
    c=make_clean()
    Image.fromarray(c.astype(np.uint8)).save('clean.png')
    m=glass_mask()
    ov=orig.copy(); ov[...,0]=ov[...,0]*(1-0.5*m)+255*0.5*m
    Image.fromarray(ov.astype(np.uint8)).crop((100,120,1150,880)).save('mask_ov.png')
    Image.fromarray(c.astype(np.uint8)).crop((100,120,1150,880)).save('clean_crop.png')
