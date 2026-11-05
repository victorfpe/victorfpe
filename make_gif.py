#!/usr/bin/env python3
"""Gera retro-tv.gif: TV CRT original intacta + 6 canais animados dentro do vidro."""
import sys, numpy as np, cv2
from PIL import Image
from base import make_clean, glass_mask, H, W

OUT = sys.argv[1] if len(sys.argv) > 1 else 'retro-tv.gif'
OUT_W = int(sys.argv[2]) if len(sys.argv) > 2 else 800
rng = np.random.default_rng(2026)

# ---------- fonte pixel 5x7 ----------
G = {
'A':"01110 10001 10001 11111 10001 10001 10001",'B':"11110 10001 10001 11110 10001 10001 11110",
'C':"01110 10001 10000 10000 10000 10001 01110",'D':"11110 10001 10001 10001 10001 10001 11110",
'E':"11111 10000 10000 11110 10000 10000 11111",'F':"11111 10000 10000 11110 10000 10000 10000",
'G':"01110 10001 10000 10111 10001 10001 01111",'H':"10001 10001 10001 11111 10001 10001 10001",
'I':"01110 00100 00100 00100 00100 00100 01110",'J':"00111 00010 00010 00010 00010 10010 01100",
'K':"10001 10010 10100 11000 10100 10010 10001",'L':"10000 10000 10000 10000 10000 10000 11111",
'M':"10001 11011 10101 10101 10001 10001 10001",'N':"10001 11001 10101 10011 10001 10001 10001",
'O':"01110 10001 10001 10001 10001 10001 01110",'P':"11110 10001 10001 11110 10000 10000 10000",
'Q':"01110 10001 10001 10001 10101 10010 01101",'R':"11110 10001 10001 11110 10100 10010 10001",
'S':"01111 10000 10000 01110 00001 00001 11110",'T':"11111 00100 00100 00100 00100 00100 00100",
'U':"10001 10001 10001 10001 10001 10001 01110",'V':"10001 10001 10001 10001 10001 01010 00100",
'W':"10001 10001 10001 10101 10101 11011 10001",'X':"10001 10001 01010 00100 01010 10001 10001",
'Y':"10001 10001 01010 00100 00100 00100 00100",'Z':"11111 00001 00010 00100 01000 10000 11111",
'0':"01110 10001 10011 10101 11001 10001 01110",'1':"00100 01100 00100 00100 00100 00100 01110",
'2':"01110 10001 00001 00010 00100 01000 11111",'3':"11110 00001 00001 01110 00001 00001 11110",
'4':"00010 00110 01010 10010 11111 00010 00010",'5':"11111 10000 11110 00001 00001 10001 01110",
'6':"00110 01000 10000 11110 10001 10001 01110",'7':"11111 00001 00010 00100 01000 01000 01000",
'8':"01110 10001 10001 01110 10001 10001 01110",'9':"01110 10001 10001 01111 00001 00010 01100",
'.':"00000 00000 00000 00000 00000 01100 01100",'!':"00100 00100 00100 00100 00100 00000 00100",
'>':"10000 01000 00100 00010 00100 01000 10000",'_':"00000 00000 00000 00000 00000 00000 11111",
' ':"00000 00000 00000 00000 00000 00000 00000",
'•':"00000 00000 01110 01110 01110 00000 00000",
'▶':"10000 11000 11100 11110 11100 11000 10000",
}
G = {k:[[int(c) for c in row] for row in v.split()] for k,v in G.items()}

def text_masks(text, cx, cy, cell, adv=6.0, bold=0.0, align='center', width_px=None):
    """Retorna (mask_texto, mask_bullets) float32 HxW desenhadas como pixels quadrados."""
    n = len(text); total = (n*adv - (adv-5))*cell
    x0 = cx - total/2 if align=='center' else (cx if align=='left' else cx-total)
    y0 = cy - 3.5*cell
    m = np.zeros((H,W),np.uint8); b = np.zeros((H,W),np.uint8)
    for i,ch in enumerate(text):
        tgt = b if ch in '•▶' else m
        for r,row in enumerate(G[ch]):
            for c,v in enumerate(row):
                if v:
                    xa=int(round(x0+(i*adv+c)*cell)); xb=int(round(x0+(i*adv+c+1+bold)*cell))
                    ya=int(round(y0+r*cell)); yb=int(round(y0+(r+1)*cell))
                    tgt[ya:yb, xa:max(xb,xa+1)] = 255
    return m.astype(np.float32)/255, b.astype(np.float32)/255, (y0, 7*cell)

# ---------- curvatura do vidro (distorcao em barril) ----------
CX,CY,A_,B_ = 618,503,455,336
yy,xx = np.mgrid[0:H,0:W].astype(np.float32)
r2 = ((xx-CX)/A_)**2 + ((yy-CY)/B_)**2
K = 0.11
MAPX = (CX + (xx-CX)*(1+K*r2)).astype(np.float32)
MAPY = (CY + (yy-CY)*(1+K*r2)).astype(np.float32)
def warp(img): return cv2.remap(img, MAPX, MAPY, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

SCAN = np.where(np.arange(H)%4==3, 0.50, 1.0).astype(np.float32)[:,None,None]
def vgrad(y0,h,top,bot):
    t = np.clip((np.arange(H)-y0)/max(h,1),0,1)[:,None,None]
    return (np.array(top,np.float32)*(1-t)+np.array(bot,np.float32)*t)/255.

CYAN=(130,240,255); WHITE_C=(215,250,255); LAV=(205,150,255); PURPLE=(150,50,235)
MAG=(255,90,230); GREEN=(80,255,120)

def shift(m,dy,dx): return np.roll(np.roll(m,dy,0),dx,1)

def build_layer(ch, title, sub):
    A = np.zeros((H,W,3),np.float32)
    # titulo
    n=len(title); cell=min(13.0, 800/(n*6.3-1.3))
    m,_,(ty,th) = text_masks(title, 632, 348, cell, adv=6.3, bold=0.35)
    sh = shift(m, int(cell*0.32)+1, int(cell*0.32)+1)
    A += sh[...,None]*np.array(PURPLE,np.float32)/255*0.85
    A = np.maximum(A, m[...,None]*vgrad(ty,th,WHITE_C,LAV))
    # subtitulo
    n=len(sub); cell=min(4.25, 770/(n*6-1))
    m,b,(sy,sh_) = text_masks(sub, 636, 430, cell, adv=6.0, bold=0.25)
    shd = shift(m+b,2,2)
    A += shd[...,None]*np.array(PURPLE,np.float32)/255*0.7
    A = np.maximum(A, m[...,None]*np.array(CYAN,np.float32)/255)
    A = np.maximum(A, b[...,None]*np.array(MAG,np.float32)/255)
    # indicador de canal (CH 0N + triangulo) e PLAY
    m,b,_ = text_masks(f"CH 0{ch}", 226, 242, 4.5, align='left', bold=0.2)
    tri,_,_ = text_masks("", 0,0,1) if False else (None,None,None)
    A = np.maximum(A, m[...,None]*np.array(CYAN,np.float32)/255)
    _,b,_ = text_masks("▶", 226+ (6*6-1)*4.5 + 6, 242, 4.5, align='left')
    A = np.maximum(A, b[...,None]*np.array(WHITE_C,np.float32)/255)
    m,_,_ = text_masks("PLAY", 1002, 242, 4.5, align='right', bold=0.2)
    A = np.maximum(A, m[...,None]*np.array(GREEN,np.float32)/255)
    # pips de canal (6 quadradinhos)
    pm = np.zeros((H,W,3),np.float32)
    for i in range(6):
        x = 632 + (i-2.5)*22
        col = np.array(CYAN if i+1==ch else (90,60,150),np.float32)/255 * (1.0 if i+1==ch else 0.8)
        pm[236:248, int(x-5):int(x+5)] = col
    A = np.maximum(A, pm)
    A = warp(A)
    glow = cv2.GaussianBlur(A,(0,0),4.0)*0.9 + cv2.GaussianBlur(A,(0,0),13)*0.7
    return np.clip(A*SCAN + glow, 0, 1.6)

def build_cursor():
    c = np.zeros((H,W,3),np.float32)
    c[781:791,764:790] = np.array((170,255,215),np.float32)/255
    c = warp(c)
    return np.clip(c*SCAN + cv2.GaussianBlur(c,(0,0),5)*1.0, 0, 1.6)

CHANNELS = [
 ("VICTOR.DEV","WELCOME TO MY PORTFOLIO"),
 ("ABOUT ME","TECHNOLOGY • CURIOSITY • LEARNING"),
 ("SKILLS","JAVA • HTML • CSS • JAVASCRIPT"),
 ("CYBERSECURITY","LINUX • NETWORKS • SECURITY"),
 ("PROJECTS","DEVJOBS • RETRO TV • MORE SOON"),
 ("TRANSMISSION","SYSTEM ONLINE • THANKS FOR VISITING"),
]
base = make_clean()
mask = glass_mask(shrink=8)[...,None]
layers = [build_layer(i+1,*c) for i,c in enumerate(CHANNELS)]
cursor = build_cursor()

# ---------- efeitos de quadro ----------
def noise_img():
    small = rng.random((H//12, W//18)).astype(np.float32)
    n = cv2.resize(small,(W,H),interpolation=cv2.INTER_NEAREST)
    n = n*(0.55+0.45*(np.arange(H)%3!=0)[:,None])
    return n[...,None]*np.array([150,215,255],np.float32)/255*255

def tear(img, amp):
    out = img.copy(); y=170
    while y<840:
        h=int(rng.integers(5,40)); dx=int(rng.normal(0,amp))
        if dx: out[y:y+h]=np.roll(img[y:y+h],dx,1)
        y+=h
    return out

def frame(layer=None, alpha=1.0, gain=1.0, noise=0.0, tear_amp=0, cursor_on=False, band=None):
    img = base.copy()
    if layer is not None: img = img + layer*255*alpha
    if cursor_on: img = img + cursor*255*alpha
    img = np.clip(img,0,255)
    if tear_amp: img = tear(img, tear_amp)
    if noise:  img = img*(1-noise*0.85*mask) + noise_img()*noise*0.85*mask*0.7
    if band is not None:
        y=band; prof=np.exp(-((np.arange(H)-y)/22.0)**2)[:,None,None]*0.07
        img = img*(1+prof*mask)
    img = img*(1+(gain-1)*mask)
    # a transicao so mexe dentro do vidro: fora da mascara permanece a TV original
    img = base*(1-mask)+img*mask if (tear_amp or noise) else img
    return np.clip(img,0,255).astype(np.uint8)

frames=[]; durs=[]
def add(f,d): frames.append(f); durs.append(d)

for i in range(6):
    L = layers[i]; N = layers[(i+1)%6]
    # exibicao do canal com flicker discreto e cursor piscando
    add(frame(L, cursor_on=True), 700)
    add(frame(L, gain=1.06, cursor_on=True, band=330), 70)
    add(frame(L, cursor_on=False), 700)
    add(frame(L, cursor_on=True), 700)
    add(frame(L, gain=1.04, cursor_on=False, band=690), 60)
    add(frame(L, cursor_on=False), 550)
    # transicao suave entre canais (estatica + rasgos horizontais)
    add(frame(L, 0.50, noise=0.38, tear_amp=22, cursor_on=True), 70)
    add(frame(None,0, noise=0.70, tear_amp=40, gain=1.1), 80)
    add(frame(N, 0.55, noise=0.36, tear_amp=18, cursor_on=True), 70)
    add(frame(N, 0.92, noise=0.10, tear_amp=5,  cursor_on=True), 70)

# ---------- redimensiona + paleta global + GIF ----------
OUT_H = round(H*OUT_W/W)
small=[Image.fromarray(f).resize((OUT_W,OUT_H),Image.LANCZOS) for f in frames]
sample = [small[k] for k in range(0,len(small),3)]
mosaic = Image.new('RGB',(OUT_W, OUT_H*len(sample)))
for k,s in enumerate(sample): mosaic.paste(s,(0,k*OUT_H))
pal = mosaic.quantize(colors=160, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
q=[s.quantize(palette=pal, dither=Image.Dither.NONE) for s in small]
q[0].save(OUT, save_all=True, append_images=q[1:], duration=durs, loop=0, optimize=False, disposal=1)
import os
print(OUT, f"{OUT_W}x{OUT_H}", len(frames),'frames', round(os.path.getsize(OUT)/1e6,2),'MB', sum(durs)/1000,'s')
# previews
Image.fromarray(frames[0]).save('prev_ch1.png'); Image.fromarray(frames[7*3+5*1+0]).save('prev_tmp.png')
