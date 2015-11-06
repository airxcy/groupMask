//
//  klt_gpu.h
//  ITF_Inegrated
//
//  Created by Chenyang Xia on 8/18/2015.
//  Copyright (c) 2015 CUHK. All rights reserved.
//

#ifndef KLTTRACKER_H
#define KLTTRACKER_H
#include <string>
#include <vector>
#include "trackers/buffers.h"
#include "trackers/utils.h"
#include <opencv2/core/core.hpp>
class KLTsparse_CUDA
{
public:
    KLTsparse_CUDA();
    ~KLTsparse_CUDA();
    int init(int w,int h,unsigned char* framedata,int nPoints);
    int selfinit(unsigned char* framedata);
    int updateAframe(unsigned char* framedata,int fidx);
    int endTraking();

    /** Basic **/
    int frame_width=0, frame_height=0;
    int frameidx=0;
    bool persDone=false,render=true;
    void setUpPersMap(float *srcMap);
    void updateROICPU(float* aryPtr,int length);
    float * h_persMap;

    /** Point Tracking and Detecting **/
    int nFeatures=0,nSearch=0;
    FeatPts pttmp;
    std::vector<FeatBuff> trackBuff;
    float* h_curvec,* h_corners;
    bool applyseg=false;
    unsigned char* h_roimask;
    void updateSegCPU(unsigned char* ptr);
    void updateSegNeg(float* aryPtr,int length);
    bool checkTrackMoving(FeatBuff &strk);
    void PointTracking();
    void findPoints();
    void filterTrack();

    /** Grouping **/
    bool groupOnFlag=true;
    ofv ofvtmp;
    Buff<ofv> ofvBuff;
    int offsetidx=0;
    unsigned char* h_neighborD,* h_clrvec,* h_curnbor;
    int *h_KnnIdx;
    float *h_distmat;
    void knn();
    int curK=0,pregroupN=0,groupN=0,maxgroupN=0;
    std::vector<int> items;
    void bfsearch();
    void reGroup();

    /** Group Properties **/
    int* viewed;
    int * h_gAge;
    bool updateFlag=false,calPolyGon=true;
    int curTrkingIdx=0;
    FeatBuff bbTrkBUff;
    int *h_prelabel,*h_label,*label_final,*h_gcount,*h_overlap;
    float *h_com,*h_precom,*h_bbox,*h_bbstats;
    cvxPnt cvxPnttmp;
    std::vector< std::vector<cvxPnt> > setPts;
    std::vector<FeatBuff> cvxPts;
    std::vector<TrackBuff> cvxInt;

    /** Render Frame **/
    int zoomW=255,zoomH=255;
    unsigned char* h_zoomFrame=NULL;
    void Render(unsigned char * framedata);
};
#endif
