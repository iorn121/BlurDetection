import numpy as np
import cv2
from decimal import *
import os, glob
import sys
from pathlib import Path

# ウェーブレット変換
def wavelet(img,level):

    h,w=img.shape
    h=int(h/(2**(level-1)))
    w=int(w/(2**(level-1)))
    
    # 縦横の半分
    h_half=int(h/2)
    w_half=int(w/2)

    # imgをコピー
    cimg=img.astype(np.float64)

    # 各行ごとに処理
    for i in range(h):
        for j in range(0,w_half):
            a=int(img[i,j*2])
            b=int(img[i,2*j+1])
            cimg[i,j]=(a+b)/2.0
            cimg[i,w_half+j]=abs(1.0*a-b)/2

    # imgをコピー
    cimg2=cimg.astype(np.float64)

    # 各列ごとに処理
    for j in range(w):
        for i in range(0,h_half):
            a=int(cimg[i*2,j])
            b=int(cimg[i*2+1,j])
            cimg2[i,j]=(a+b)/2.0
            cimg2[h_half+i,j]=abs(1.0*a-b)/2

    # 戻り値
    return cimg2

# エッジの検出
def edgeanalysis(img,level):

    # 縦横ピクセル数の取得
    h,w=img.shape
    step=2**level

    # エッジマップ作成
    xh=int(h/(2**(4-level)))
    xw=int(w/(2**(4-level)))
    emap=np.zeros((xh,xw))
    for i in range(xh):
        for j in range(xw):
            emap[i,j]=(img[xh+i,j]**2+img[i,xw+j]**2+img[xh+i,xw+j]**2)**(1/2)

    # 最大値マップ作成
    yh=int(h/16)
    yw=int(w/16)
    emax=np.zeros((yh,yw))

    # 各行ごとに処理
    for i in range(yh):
        # 各列ごとに処理
        for j in range(yw):
            #ウィンドウの切り出しと最大値検出
            temp=emap[step*i:step*(i+1),step*j:step*(j+1)]
            emax[i,j]=temp.max()

    # emaxを戻り値として返す
    return emax

def countedge(img1,img2,img3,thre):

    # ラベルマップを作成
    h,w=img1.shape
    labelmap=np.zeros(((h,w,3)))

    # ラベリング用の色設定
    red = (0,0,255)
    blue = (255,0,0)
    green = (0,255,0)
    white = (255,255,255)
    black = (0,0,0)

    # 各ラベル数を数える
    Nedge=0
    Nda=0
    Nrg=0
    Nbrg=0

    for i in range(h):
        for j in range(w):
            # エッジの部分
            if img1[i,j]>thre or img2[i,j]>thre or img3[i,j]>thre:
                labelmap[i,j]=black
                Nedge+=1
                # D, Aの検出
                if img1[i,j]>img2[i,j]>img3[i,j]:
                    labelmap[i,j]=red
                    Nda+=1
                # R, Gの検出
                if img1[i,j]<img2[i,j]<img3[i,j]:
                    labelmap[i,j]=green
                    Nrg+=1
                    # 消失したR, Gの検出
                    if img1[i,j]<thre:
                        labelmap[i,j]=blue
                        Nbrg+=1
                # Rの検出
                if img1[i,j]<img2[i,j] and img2[i,j]>img3[i,j]:
                    labelmap[i,j]=green
                    Nrg+=1
                    # 消失したR, Gの検出
                    if img1[i,j]<thre:
                        labelmap[i,j]=blue
                        Nbrg+=1
            # エッジ以外の部分
            else:
                labelmap[i,j]=white
    
    # ブレ判定の指標
    if Nedge!=0:
        Per=1.0*Nda/Nedge
    else:
        Per=0
    if Nrg!=0:
        BlurExtent=1.0*Nbrg/Nrg
    else:
        BlurExtent=0

    

    # ラベルマップの戻し
    return labelmap, Per, BlurExtent


def main():
    # ファイル取得
    files = glob.glob("./input/*.jpg")

    result = "全画像の分析結果\n\n"

    for file in files:

        name = Path(file).stem

        # 画像の読み込み
        img_bgr=cv2.imread(file)

        # グレースケールに変換
        img_gray=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY)

        # 縦横ピクセル数取得
        height, width = img_gray.shape

        # 16の倍数か調べる
        if height%16!=0 or width%16!=0:
            print("{}-STOP".format(name))
        else:
            # ウェーブレット変換
            img1_converted=wavelet(img_gray,1)
            img2_converted=wavelet(img1_converted,2)
            img3_converted=wavelet(img2_converted,3)
            cv2.imwrite("./output/{}_converted.jpg".format(name),img3_converted)

            # エッジマップ作成
            emax1=edgeanalysis(img3_converted,1)
            emax2=edgeanalysis(img3_converted,2)
            emax3=edgeanalysis(img3_converted,3)
            
            # ラベルマップ作成
            labelmap, Per, BlurExtent=countedge(emax1,emax2,emax3,25)

            # ブレ判定
            path_w = "output/result.txt"

            result+="{}\nedge度数：".format(name)+str(Per)+"\n"
            if Per>0.05:
                result+="blur無"
            else:
                result+="blur有"

            # ブレの強度
            result+= "\nblur強度："+ str(BlurExtent) + "\n\n"

            cv2.imwrite("./output/{}_edgemap.jpg".format(name),labelmap)

    #出力
    with open(path_w, mode="w") as f:
        f.write(result)

if __name__ == "__main__":
    main()