#include "Qts/viewqt.h"
#include "Qts/modelsqt.h"
#include "Qts/streamthread.h"

#include <iostream>
#include <stdio.h>

#include <QPainter>
#include <QBrush>
#include <QPixmap>
#include <cmath>
#include <QGraphicsSceneEvent>
#include <QMimeData>
#include <QByteArray>
#include <QFont>
char viewstrbuff[200];
QPointF points[100];

void DefaultScene::mousePressEvent ( QGraphicsSceneMouseEvent * event )
{
    emit clicked(event);
}
void DefaultScene::drawBackground(QPainter * painter, const QRectF & rect)
{
    QPen pen;
    QFont txtfont("Roman",40);
    txtfont.setBold(true);
    pen.setColor(QColor(255,255,255));
    pen.setCapStyle(Qt::RoundCap);
    pen.setJoinStyle(Qt::RoundJoin);
    pen.setWidth(10);
    painter->setPen(QColor(243,134,48,150));
    painter->setFont(txtfont);
    painter->drawText(rect, Qt::AlignCenter,"打开文件\nOpen File");
}
TrkScene::TrkScene(const QRectF & sceneRect, QObject * parent):QGraphicsScene(sceneRect, parent)
{
    streamThd=NULL;
}
TrkScene::TrkScene(qreal x, qreal y, qreal width, qreal height, QObject * parent):QGraphicsScene( x, y, width, height, parent)
{
    streamThd=NULL;
}


void TrkScene::drawBackground(QPainter * painter, const QRectF & rect)
{
    //std::cout<<streamThd->inited<<std::endl;
    if(streamThd!=NULL&&streamThd->inited)
    {
        updateFptr(streamThd->frameptr, streamThd->frameidx);
    }
    painter->setBrush(bgBrush);
    painter->drawRect(rect);
    painter->setBrush(Qt::NoBrush);
    if(streamThd!=NULL&&streamThd->inited)
    {

        std::vector<FeatBuff>& klttrkvec=streamThd->tracker->trackBuff;

        int nFeatures= streamThd->tracker->nFeatures;
        int* labelvec=streamThd->tracker->label_final;
        unsigned char* clrvec=streamThd->tracker->h_clrvec;
        int groupN = streamThd->tracker->maxgroupN;
        int* gcount =streamThd->tracker->h_gcount;
        float* comvec = streamThd->tracker->h_com;
        float* bboxvec=streamThd->tracker->h_bbox;
        int nSearch=streamThd->tracker->nSearch;
        int* gAge=streamThd->tracker->h_gAge;
        int curTrking=streamThd->tracker->curTrkingIdx;
        unsigned char* h_neigbor=streamThd->tracker->h_neighborD;
        linepen.setColor(QColor(255,200,200,255));
        linepen.setWidth(4);
        painter->setPen(linepen);
        painter->setFont(QFont("System",10,4));
        QString infoString="fps:"+QString::number(streamThd->fps)+"\n"
                +"curTrking:"+QString::number(curTrking)+"\n"
                +"groupN:"+QString::number(groupN)+"\n";
        painter->drawText(rect, Qt::AlignLeft|Qt::AlignTop,infoString);
        painter->setFont(QFont("System",20,2));

        float x0,y0,x1,y1;

        for(int i=0;i<klttrkvec.size();i++)
        {
            FeatBuff& klttrk= klttrkvec[i];
            x1=klttrk.cur_frame_ptr->x,y1=klttrk.cur_frame_ptr->y;
            unsigned char r =  clrvec[labelvec[i]*3],g = clrvec[labelvec[i]*3+1],b = clrvec[labelvec[i]*3+2];
            linepen.setColor(QColor(r, g, b,70));
            linepen.setWidth(0);
            painter->setPen(linepen);
            painter->drawPoint(x1,y1);
            painter->setFont(QFont("System",10,0));
            for (int j = i+1; j < nFeatures; j++)
            {
                int xj = klttrkvec[j].cur_frame_ptr->x, yj = klttrkvec[j].cur_frame_ptr->y;
                    if(h_neigbor[i*nFeatures+j])
                    {
                        painter->setPen(linepen);
                        painter->drawLine(x1, y1, xj, yj);
                        //painter->drawText(xmid,ymid,QString::number(dist));
                    }
            }
        }
        for(int i=0;i<groupN;i++)
        {
            if(gcount[i]>0)
            {
                /*
                unsigned char r=255,g=255,b=255;

                //r=clrvec[i*3],g=clrvec[i*3+1],b=clrvec[i*3+2];
                TrackBuff& cvx = streamThd->tracker->cvxInt[i];
                //FeatBuff& cvx = streamThd->tracker->cvxPts[i];
                linepen.setColor(QColor(r, g, b));
                linepen.setWidth(2);
                painter->setPen(linepen);
                int x0,y0,x1,y1;

                for(int j=1;j<cvx.len;j++)
                {
                    x0=cvx.getPtr(j-1)->x,y0=cvx.getPtr(j-1)->y;
                    x1=cvx.getPtr(j)->x,y1=cvx.getPtr(j)->y;
                    painter->drawLine(x0,y0,x1,y1);

                }
                linepen.setColor(QColor(0, 255, 255));
                painter->setPen(linepen);
                painter->setBrush(Qt::NoBrush);
                painter->drawRect(QRect(bboxvec[i*4],bboxvec[i*4+1],bboxvec[i*4+2],bboxvec[i*4+3]));
                */
                linepen.setColor(QColor(255, 255, 0));
                painter->setPen(linepen);
                painter->drawText(comvec[i*2],comvec[i*2+1],QString::number(i)+":"+QString::number(gAge[i]));
            }
        }

        float* h_corners = streamThd->tracker->h_corners;
        for(int i=0;i<nSearch;i++)
        {
            linepen.setColor(QColor(255,0,0));
            linepen.setWidth(2);
            int x=h_corners[i*2]+0.5,y=h_corners[i*2+1]+0.5;
            float val = (i+0.1)/(nSearch+0.1)*255.0;
            linepen.setColor(QColor(255,255,255,255-val));
            painter->setPen(linepen);
            painter->drawPoint(x,y);
        }


        FeatBuff& bbTrkBUff =streamThd->tracker->bbTrkBUff;
        if(int i=streamThd->tracker->curTrkingIdx)
        {
            painter->setPen(QPen(QColor(255,255,255),4));
            //int x = comvec[i*2],y=comvec[i*2+1];
            QPoint pt1,pt2;
            pt1.rx()=bboxvec[i*4],pt1.ry()=bboxvec[i*4+1];
            pt2.rx()=bboxvec[i*4]+bboxvec[i*4+2],pt2.ry()=bboxvec[i*4+1]+bboxvec[i*4+3];

            linepen.setColor(QColor(255,255,255,150));
            linepen.setWidth(1);
            painter->setPen(linepen);
            painter->drawRect(QRectF(pt1,pt2));
            float alpha=0.5;
            unsigned char r=clrvec[i*3],g=clrvec[i*3+1],b=clrvec[i*3+2];
            painter->setPen(QPen(QColor(r,g,b),1));
            painter->drawLine(0,bboxvec[i*4+1]+bboxvec[i*4+3]/2.0,bboxvec[i*4],bboxvec[i*4+1]+bboxvec[i*4+3]/2.0);
            painter->drawLine(width()-1,bboxvec[i*4+1]+bboxvec[i*4+3]/2.0,bboxvec[i*4]+bboxvec[i*4+2],bboxvec[i*4+1]+bboxvec[i*4+3]/2.0);
            painter->drawLine(bboxvec[i*4]+bboxvec[i*4+2]/2.0,0,bboxvec[i*4]+bboxvec[i*4+2]/2.0,bboxvec[i*4+1]);
            painter->drawLine(bboxvec[i*4]+bboxvec[i*4+2]/2.0,height()-1,bboxvec[i*4]+bboxvec[i*4+2]/2.0,bboxvec[i*4+1]+bboxvec[i*4+3]);
            for(int j=1;j<bbTrkBUff.len;j++)
            {
                pt2.rx()=bbTrkBUff.getPtr(j)->x,pt2.ry()=bbTrkBUff.getPtr(j)->y;
                pt1.rx()=bbTrkBUff.getPtr(j-1)->x,pt1.ry()=bbTrkBUff.getPtr(j-1)->y;
                painter->drawLine(pt1,pt2);
            }

        }
    }

    QImage img (streamThd->tracker->h_zoomFrame,streamThd->tracker->zoomW,streamThd->tracker->zoomH
            ,streamThd->tracker->zoomW*3,QImage::Format_RGB888);
    painter->setBrush(img);
    int k = int(width())%int(streamThd->tracker->zoomW);
    painter->setBrushOrigin(k,0);
    painter->drawRect(QRect(width()-streamThd->tracker->zoomW,0,streamThd->tracker->zoomW,streamThd->tracker->zoomH));
    //update();
    //views().at(0)->update();
}
void TrkScene::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    if(event->button()==Qt::RightButton)
    {
        int x = event->scenePos().x(),y=event->scenePos().y();
        DragBBox* newbb = new DragBBox(x-10,y-10,x+10,y+10);
        int pid = dragbbvec.size();
        newbb->bbid=pid;
        newbb->setClr(255,255,255);
        sprintf(newbb->txt,"%c\0",pid+'A');
        dragbbvec.push_back(newbb);
        addItem(newbb);
    }
    QGraphicsScene::mousePressEvent(event);
}
void TrkScene::updateFptr(unsigned char * fptr,int fidx)
{
    bgBrush.setTextureImage(QImage(fptr,streamThd->framewidth,streamThd->frameheight,QImage::Format_RGB888));
    frameidx=fidx;
    //std::cout<<frameidx<<std::endl;
}
void TrkScene::clear()
{
    bgBrush.setStyle(Qt::NoBrush);
}
