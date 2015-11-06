#include "trackers/klt_gpu.h"

#include <cmath>
#include <fstream>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

#include <cuda.h>
#include <cublas.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include "trackers/targetfinder.h"
using namespace cv;
using namespace std;
#define PI 3.14159265

#define minDist 2
#define minGSize 1
#define TIMESPAN 15
#define COSTHRESH 0.4
#define VeloThresh 0.1
#define KnnK 40
#define MoveFactor 0.00001
#define coNBThresh 0
#define minTrkLen 2
#define Pers2Range(pers) pers/6
#define gpu_zalloc(ptr, num, size) cudaMalloc(&ptr,size*num);cudaMemset(ptr,0,size*num);

Mat prePts,nextPts,status,eigenvec, corners;

//cv::gpu::GoodFeaturesToTrackDetector_GPU* detector;
TargetFinder* detector;
cv::gpu::PyrLKOpticalFlow* tracker;
cv::gpu::GpuMat gpuGray, gpuPreGray,rgbMat,maskMat,roiMaskMat,gpuPersMap,gpuSegMat,
        gpuCorners, gpuPrePts, gpuNextPts,gpuStatus,gpuEigenvec;


typedef struct
{
    int i0, i1;
    float correlation;
}ppair, p_ppair;

//Basic
__device__ int d_framewidth[1],d_frameheight[1];
unsigned char * d_rgbframedata;
float *d_persMap;
//Rendering
__device__ unsigned char * renderMaskPtr[1];
unsigned char * d_renderMask=NULL,* d_clrvec;
int *d_label;
TrkPts* d_cvx=NULL;
int* d_cvxLen=NULL;
//Tracking
unsigned char *d_mask,*d_roimask,*d_segmask,*d_segNeg;
float* d_corners, *d_curvec;
//Grouping
unsigned char *d_neighbor,* d_neighborD;
float* d_cosine,*d_velo,* d_distmat;
ofv* d_ofvec;

int *tmpn,*idxmap;

cublasHandle_t handle;

__global__ void applyPersToMask(unsigned char* d_mask,float* d_curvec,float* d_persMap)
{
    int pidx=blockIdx.x;
    float px=d_curvec[pidx*2],py=d_curvec[pidx*2+1];
    int blocksize = blockDim.x;
    int w=d_framewidth[0],h=d_frameheight[0];
    int localx = threadIdx.x,localy=threadIdx.y;
    int pxint = px+0.5,pyint = py+0.5;
    float persval =d_persMap[pyint*w+pxint];
    float range=Pers2Range(persval);
    int offset=range+0.5;
    int yoffset = localy-blocksize/2;
    int xoffset = localx-blocksize/2;
    if(abs(yoffset)<range&&abs(xoffset)<range)
    {
        int globalx=xoffset+pxint,globaly=yoffset+pyint;
        d_mask[globaly*d_framewidth[0]+globalx]=0;
    }
}
__global__ void applySegMask(unsigned char* d_mask,unsigned char* d_segmask,unsigned char* d_segNeg)
{
    int offset=blockIdx.x*blockDim.x+threadIdx.x;
    int w=d_framewidth[0],h=d_frameheight[0];
    int totallen =w*h;
    int y=offset/w;
    int x=offset%w;
    if(offset<totallen&&!d_segNeg[offset]&&!d_segmask[offset])
    {
        d_mask[offset]=0;
    }
}
__device__ int polyTest(TrkPts* d_cvx,int total,int x,int y)
{
    int result=0;
    CvPoint v0, v;
    TrkPts ip;
    ip.x=x,ip.y=y;
    TrkPts* ptr = d_cvx;
    v.x=ptr->x,v.y=ptr->y;
    int counter=0;
    for(int i = 0; i < total; i++ )
    {
        int dist;
        v0 = v;

        ptr++;
        v.x=ptr->x,v.y=ptr->y;

        if( (v0.y <= ip.y && v.y <= ip.y) ||
            (v0.y > ip.y && v.y > ip.y) ||
            (v0.x < ip.x && v.x < ip.x) )
        {
            if( ip.y == v.y && (ip.x == v.x || (ip.y == v0.y &&
                ((v0.x <= ip.x && ip.x <= v.x) || (v.x <= ip.x && ip.x <= v0.x)))) )
                return 0;
            continue;
        }

        dist = (ip.y - v0.y)*(v.x - v0.x) - (ip.x - v0.x)*(v.y - v0.y);
        if( dist == 0 )
            return 0;
        if( v.y < v0.y )
            dist = -dist;
        counter += dist > 0;
    }
    result = (counter % 2 == 0)? -1 : 1;
    return result;
}
__global__ void renderFrame(unsigned char* d_renderMask,unsigned char* d_frameptr)
{
    int offset=blockIdx.x*blockDim.x+threadIdx.x;
    int totallen =d_frameheight[0]*d_framewidth[0];
    float alpha=0.5;
    if(offset<totallen)
    {
        d_frameptr[offset*3]=d_frameptr[offset*3]*alpha+d_renderMask[offset*3]*(1-alpha);
        d_frameptr[offset*3+1]=d_frameptr[offset*3+1]*alpha+d_renderMask[offset*3+1]*(1-alpha);
        d_frameptr[offset*3+2]=d_frameptr[offset*3+2]*alpha+d_renderMask[offset*3+2]*(1-alpha);
    }
}
__global__ void renderGroup(unsigned char* d_renderMask,float* d_curvec ,int* d_label,unsigned char* d_clrvec,float* d_persMap,TrkPts* d_cvx,int* d_cvxLen)
{
    int pidx=blockIdx.x;
    int px=d_curvec[pidx*2]+0.5,py=d_curvec[pidx*2+1]+0.5;
    int label=d_label[pidx];
    int blocksize = blockDim.x;
    int w=d_framewidth[0],h=d_frameheight[0];
    float persval =d_persMap[py*w+px];
    float range=Pers2Range(persval);
    int centerOffset=blocksize/2;
    int xoffset = threadIdx.x-centerOffset;
    unsigned char r=d_clrvec[label*3],g=d_clrvec[label*3+1],b=d_clrvec[label*3+2];
    float alpha = 0.5;
    int nFeatures = gridDim.x;
    //d_renderMask[pidx+w*threadIdx.x]=d_cvx[pidx+nFeatures*threadIdx.x].x;
    if(label)
    {

        for(int i=0;i<blocksize;i++)
        {
            int yoffset = i-centerOffset;
            if(abs(yoffset)<range&&abs(xoffset)<range)
            {
                int globalx=xoffset+px,globaly=yoffset+py;
                int offset = globaly*w+globalx;
                d_renderMask[offset*3]=d_renderMask[offset*3]*alpha+r*(1-alpha);
                d_renderMask[offset*3+1]=d_renderMask[offset*3+1]*alpha+g*(1-alpha);
                d_renderMask[offset*3+2]=d_renderMask[offset*3+2]*alpha+b*(1-alpha);
            }

            /*
            int globalx=xoffset+px,globaly=yoffset+py;
            if(globaly>0&&globaly<h&&globalx>0&&globalx<w)
            {
                int offset = globaly*w+globalx;
                int cvxlen = d_cvxLen[label];
                if(cvxlen>0)
                {
                    int offset = globaly*w+globalx;
                    TrkPts* cvxvec = d_cvx+1024*label;
                    if(polyTest(cvxvec,cvxlen,globalx,globaly)>0)
                    {
                        d_renderMask[offset*3]=d_renderMask[offset*3]*alpha+r*(1-alpha);
                        d_renderMask[offset*3+1]=d_renderMask[offset*3+1]*alpha+g*(1-alpha);
                        d_renderMask[offset*3+2]=d_renderMask[offset*3+2]*alpha+b*(1-alpha);
                    }
                }
            }
            */
        }
    }
}
__global__ void searchNeighborMap(unsigned char* d_neighbor,float* d_cosine,float* d_velo, ofv* d_ofvec,float* d_distmat ,float * d_persMap, int nFeatures)
{
    int r = blockIdx.x, c = threadIdx.x;
    if (r < c)
    {
        unsigned char* curptr=d_neighbor;
        float dx = abs(d_ofvec[r].x1 - d_ofvec[c].x1), dy = abs(d_ofvec[r].y1 - d_ofvec[c].y1);
        int yidx = d_ofvec[r].idx, xidx = d_ofvec[c].idx;
        int  ymid = (d_ofvec[r].y1 + d_ofvec[c].y1) / 2.0+0.5,xmid = (d_ofvec[r].x1 + d_ofvec[c].x1) / 2+0.5;
        float persval=0;
        //if(ymid<d_frameheight[0]&&xmid<d_framewidth[0])
        persval =d_persMap[ymid*d_framewidth[0]+xmid];
        float hrange=persval,wrange=persval;

        if(hrange<2)hrange=2;
        if(wrange<2)wrange=2;
        if (dx < wrange && dy < hrange)
        {
            curptr[yidx*nFeatures+xidx]=1;
            curptr[xidx*nFeatures+yidx]=1;
            float vx0 = d_ofvec[r].x1 - d_ofvec[r].x0, vx1 = d_ofvec[c].x1 - d_ofvec[c].x0,
                vy0 = d_ofvec[r].y1 - d_ofvec[r].y0, vy1 = d_ofvec[c].y1 - d_ofvec[c].y0;
            float dist = dx*dx+dy*dy;
            float norm0 = sqrt(vx0*vx0 + vy0*vy0), norm1 = sqrt(vx1*vx1 + vy1*vy1);
            float cosine = (vx0*vx1 + vy0*vy1) / norm0 / norm1;

            d_cosine[yidx*nFeatures+xidx]=cosine;
            d_cosine[xidx*nFeatures+yidx]=cosine;
            d_distmat[yidx*nFeatures+xidx]=dist;
            d_distmat[xidx*nFeatures+yidx]=dist;
            /*
            d_velo[(yidx*nFeatures+xidx)*2]=vx0;
            d_velo[(yidx*nFeatures+xidx)*2+1]=vy0;
            d_velo[(xidx*nFeatures+yidx)*2]=vx1;
            d_velo[(xidx*nFeatures+yidx)*2+1]=vy1;
            */
        }
    }
}
__global__ void neighborD(unsigned char* d_neighbor,float* d_cosine,float* d_velo,unsigned char* d_neighborD,int offsetidx)
{
    int yidx = blockIdx.x, xidx = threadIdx.x,nFeatures = blockDim.x;
    unsigned char val = 1;
    float cosine = 0,norm0 =0,norm1=0,vx0=0,vy0=0,vx1=0,vy1=0;
    bool cosflag=true;
    for(int i=0;i<TIMESPAN;i++)
    {
        val=val&&d_neighbor[i*nFeatures*nFeatures+yidx*nFeatures+xidx];
        cosflag=cosflag&&d_cosine[i*nFeatures*nFeatures+yidx*nFeatures+xidx];
        cosine+=d_cosine[i*nFeatures*nFeatures+yidx*nFeatures+xidx];
        /*
        vx0+=d_velo[(i*nFeatures*nFeatures+yidx*nFeatures+xidx)*2];
        vy0+=d_velo[(i*nFeatures*nFeatures+yidx*nFeatures+xidx)*2+1];
        vx1+=d_velo[(i*nFeatures*nFeatures+xidx*nFeatures+yidx)*2];
        vy1+=d_velo[(i*nFeatures*nFeatures+xidx*nFeatures+yidx)*2+1];
        */
    }
    cosine/=(TIMESPAN+1);
    /*
    vx0/=(TIMESPAN+1);
    vy0/=(TIMESPAN+1);
    vx1/=(TIMESPAN+1);
    vy1/=(TIMESPAN+1);
    norm0 = sqrt(vx0*vx0 + vy0*vy0), norm1 = sqrt(vx1*vx1 + vy1*vy1);
    float veloVar= abs(norm0-norm1)/(norm0+norm1);
    */
    float veloVar=0;
    if(val&&cosine>COSTHRESH&&veloVar<VeloThresh)
    //if(val&&cosflag)
        d_neighborD[yidx*nFeatures+xidx]=1;
}

KLTsparse_CUDA::KLTsparse_CUDA()
{
    frame_width=0, frame_height=0;
    frameidx=0;
    nFeatures=0,nSearch=0; 
    /**cuda **/
    persDone=false;
}
KLTsparse_CUDA::~KLTsparse_CUDA()
{
    tracker->releaseMemory();
    detector->releaseMemory();
    gpuGray.release();
    gpuPreGray.release();
    rgbMat.release();
    gpuCorners.release(); 
    gpuPrePts.release();
    gpuNextPts.release();
    gpuStatus.release();
    gpuEigenvec.release();
    cudaFree(d_curvec);
}
int KLTsparse_CUDA::init(int w, int h,unsigned char* framedata,int nPoints)
{
    /** Checking Device Properties **/
    int nDevices;
    int maxthread=0;
    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        /*
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n",prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n",prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n\n",2.0*prop.memoryClockRate*(prop.memoryBusWidth / 8) / 1.0e6);
        std::cout << "maxgridDim" << prop.maxGridSize[0] << "," << prop.maxGridSize[1] << "," << prop.maxGridSize[2] << std::endl;
        std::cout<<"maxThreadsPerBlock:"<<prop.maxThreadsPerBlock<<std::endl;
        */

        //cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize,MyKernel, 0, arrayCount);
        if(maxthread==0)maxthread=prop.maxThreadsPerBlock;
        //std::cout << prop.major << "," << prop.minor << std::endl;
    }


    /** Basic **/
    frame_width = w,frame_height = h;
    frameidx=0;
    cudaMemcpyToSymbol(d_framewidth,&frame_width,sizeof(int));
    cudaMemcpyToSymbol(d_frameheight,&frame_height,sizeof(int));
    gpuGray=gpu::GpuMat(frame_height, frame_width, CV_8UC1 );
    gpuPreGray=gpu::GpuMat(frame_height, frame_width, CV_8UC1 );
    gpu_zalloc(d_rgbframedata,frame_height*frame_width*3,sizeof(unsigned char));
    rgbMat = gpu::GpuMat(frame_height, frame_width, CV_8UC3 ,d_rgbframedata);
    gpu_zalloc(d_persMap, frame_width*frame_height, sizeof(float));
    gpuPersMap= gpu::GpuMat(frame_height, frame_width, CV_32F ,d_persMap);
    h_persMap =  (float*)zalloc(frame_width*frame_height, sizeof(float));
    h_roimask =  (unsigned char *)zalloc( frame_height*frame_width,sizeof(unsigned char));
    gpu_zalloc(d_roimask,frame_height*frame_width,sizeof(unsigned char));
    roiMaskMat = gpu::GpuMat(frame_height, frame_width, CV_8UC1 ,d_roimask);


    /** Point Tracking and Detecting **/
    nFeatures = maxthread;//(maxthread>1024)?1024:maxthread;
    nFeatures = (maxthread>nPoints)?nPoints:maxthread;
    nSearch=nFeatures;
    trackBuff = std::vector<FeatBuff>(nFeatures);
    for (int i=0;i<nFeatures;i++)
    {
        trackBuff[i].init(1,100);
    }
    //detector=new  gpu::GoodFeaturesToTrackDetector_GPU(nSearch,1e-30,0,3);
    detector =new TargetFinder(nSearch,1e-30,0,3);
    tracker =new  gpu::PyrLKOpticalFlow();
    tracker->winSize=Size(9,9);
    tracker->maxLevel=3;
    tracker->iters=10;
    h_corners = (float*)zalloc(nFeatures*2,sizeof(float));
    corners = Mat(1,nSearch,CV_32FC2,h_corners);
    gpu_zalloc(d_corners,2*nSearch,sizeof(float));
    gpuCorners=gpu::GpuMat(1, nSearch, CV_32FC2,d_corners);
    gpu_zalloc(d_mask,frame_height*frame_width,sizeof(unsigned char));
    maskMat = gpu::GpuMat(frame_height, frame_width, CV_8UC1 ,d_mask);
    gpu_zalloc(d_segmask,frame_height*frame_width,sizeof(unsigned char));
    gpuSegMat =gpu::GpuMat(frame_height, frame_width, CV_8UC1 ,d_segmask);
    gpu_zalloc(d_segNeg,frame_height*frame_width,sizeof(unsigned char));
    h_curvec = (float*)zalloc(nFeatures*2,sizeof(float));
    gpu_zalloc(d_curvec, nFeatures * 2, sizeof(float));

    /** Grouping **/
    //Neighbor Search
    ofvBuff.init(1, nFeatures);
    gpu_zalloc(d_neighbor,nFeatures*nFeatures*TIMESPAN,1);
    gpu_zalloc(d_cosine,nFeatures*nFeatures*TIMESPAN,sizeof(float));
    gpu_zalloc(d_velo,nFeatures*nFeatures*2*TIMESPAN,sizeof(float));
    gpu_zalloc(d_ofvec, nFeatures, sizeof(ofv));
    h_neighborD=(unsigned char*)zalloc(nFeatures*nFeatures,1);
    gpu_zalloc(d_neighborD,nFeatures*nFeatures,1);
    //k-NN
    h_KnnIdx = (int*)zalloc(KnnK,sizeof(int));
    h_curnbor = (unsigned char*)zalloc(nFeatures*nFeatures,1);
    h_distmat = (float *)zalloc(nFeatures*nFeatures,sizeof(float));
    gpu_zalloc(d_distmat,nFeatures*nFeatures,sizeof(float));
    //BFsearch
    curK=0,groupN=0;
    items.reserve(nFeatures);
    tmpn = (int*)zalloc(nFeatures,sizeof(int));
    idxmap= (int*)zalloc(nFeatures,sizeof(int));
    //label Re-Map
    maxgroupN=0;
    h_overlap = (int*)zalloc(nFeatures*nFeatures,sizeof(int));
    h_prelabel = (int*)zalloc(nFeatures,sizeof(int));
    h_label = (int*)zalloc(nFeatures,sizeof(int));
    label_final =(int*)zalloc(nFeatures,sizeof(int));
    h_gcount = (int*)zalloc(nFeatures,sizeof(int));


    /** Group Properties **/
    viewed=(int* )zalloc(nFeatures,sizeof(int));
    h_gAge=(int* )zalloc(nFeatures,sizeof(int));
    updateFlag=false;
    curTrkingIdx=0;
    //Center
    h_com = (float*)zalloc(nFeatures*2,sizeof(float));
    h_precom = (float*)zalloc(nFeatures*2,sizeof(float));
    //Bounding Box
    bbTrkBUff.init(1,125);
    h_bbox = (float*)zalloc(nFeatures*4,sizeof(float));
    h_bbstats = (float*)zalloc(nFeatures*4,sizeof(float));
    //Convex Hull
    setPts = std::vector< std::vector<cvxPnt> >(nFeatures);
    cvxPts =std::vector< FeatBuff >(nFeatures);
    cvxInt = std::vector< TrackBuff >(nFeatures);
    for(int i=0;i<nFeatures;i++)
    {
        cvxPts[i].init(1,nFeatures);
        cvxInt[i].init(1,nFeatures);
    }

    /**  render **/
    h_zoomFrame=(unsigned char *)zalloc(zoomW*zoomH*3,sizeof(unsigned char));

    gpu_zalloc(d_renderMask,frame_width*frame_height*3,sizeof(unsigned char));
    gpu_zalloc(d_label,nFeatures,sizeof(int));
    h_clrvec = (unsigned char*)zalloc(nFeatures*3,1);
    gpu_zalloc(d_clrvec,nFeatures*3,1);

    /** Self Init **/
    selfinit(framedata);
    std::cout<< "inited" << std::endl;
    return 1;
}
int KLTsparse_CUDA::selfinit(unsigned char* framedata)
{
    Mat curframe(frame_height,frame_width,CV_8UC3,framedata);
    rgbMat.upload(curframe);
    gpu::cvtColor(rgbMat,gpuGray,CV_RGB2GRAY);
    gpuGray.copyTo(gpuPreGray);
    (*detector)(gpuGray, gpuCorners);
    gpuCorners.download(corners);
    gpuCorners.copyTo(gpuPrePts);
    for (int k = 0; k < nFeatures; k++)
    {
        Vec2f p = corners.at<Vec2f>(k);
        pttmp.x = p[0];
        pttmp.y = p[1];
        pttmp.t = frameidx;
        trackBuff[k].updateAFrame(&pttmp);
        h_curvec[k * 2] = trackBuff[k].cur_frame_ptr->x;
        h_curvec[k * 2 + 1] = trackBuff[k].cur_frame_ptr->y;
    }
    cudaMemset(d_mask,255,frame_width*frame_height*sizeof(unsigned char));
    cudaMemset(d_roimask,255,frame_width*frame_height*sizeof(unsigned char));
    return true;
}
void KLTsparse_CUDA::setUpPersMap(float* srcMap)
{
    memcpy(h_persMap,srcMap,frame_width*frame_height*sizeof(float));
    cudaMemcpy(d_persMap,srcMap,frame_width*frame_height*sizeof(float),cudaMemcpyHostToDevice);
    detector->setPersMat(gpuPersMap,frame_width,frame_height);
}
bool KLTsparse_CUDA::checkTrackMoving(FeatBuff &strk)
{
    bool isTrkValid = true;
    if(strk.len>1)
    {

        PntT xb=strk.cur_frame_ptr->x,yb=strk.cur_frame_ptr->y;
        float persval = h_persMap[yb*frame_width+xb];
        PntT prex=strk.getPtr(strk.len-2)->x, prey=strk.getPtr(strk.len-2)->y;
        double trkdist=abs(prex-xb)+abs(prey-yb);
        if(trkdist>persval)return false;
        int Movelen=150/sqrt(persval),startidx=max(strk.len-Movelen,0);
        if(strk.len>Movelen)
        {
            FeatPts* aptr = strk.getPtr(startidx);
            PntT xa=aptr->x,ya=aptr->y;
            double displc=sqrt((xb-xa)*(xb-xa) + (yb-ya)*(yb-ya));
            if((strk.len -startidx)*MoveFactor>displc)
            {
                isTrkValid = false;
            }
        }


    }
    return isTrkValid;
}
void KLTsparse_CUDA::updateSegCPU(unsigned char* ptr)
{
    //Mat kernel=Mat::ones(5,5,CV_8UC1);
    cudaMemcpy(d_segmask,ptr,frame_height*frame_width,cudaMemcpyHostToDevice);
    //dilate(gpuSegMat, gpuDiaMat, kernel, Point(-3, -3));
    //cudaMemcpy(d_segmask,gpuDiaMat.data,frame_height*frame_width,cudaMemcpyHostToDevice);

}
void KLTsparse_CUDA::updateROICPU(float* aryPtr,int length)
{
    cudaMemset(d_roimask,0,frame_height*frame_width*sizeof(unsigned char));
    memset(h_roimask,0,frame_height*frame_width*sizeof(unsigned char));
    std::vector<Point2f> roivec;
    int counter=0;
    for(int i=0;i<length;i++)
    {
        Point2f p(aryPtr[i*2],aryPtr[i*2+1]);
        roivec.push_back(p);
    }
    for(int i=0;i<frame_height;i++)
    {
        for(int j=0;j<frame_width;j++)
        {
            if(pointPolygonTest(roivec,Point2f(j,i),true)>0)
            {
                h_roimask[i*frame_width+j]=255;
                counter++;

            }
        }
    }
    std::cout<<counter<<std::endl;
    cudaMemcpy(d_roimask,h_roimask,frame_height*frame_width*sizeof(unsigned char),cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask,d_roimask,frame_height*frame_width*sizeof(unsigned char),cudaMemcpyDeviceToDevice);
}
void KLTsparse_CUDA::updateSegNeg(float* aryPtr,int length)
{
    unsigned char * h_segNeg = (unsigned char *)zalloc( frame_height*frame_width,sizeof(unsigned char));
    cudaMemset(d_segNeg,0,frame_height*frame_width*sizeof(unsigned char));
    memset(h_segNeg,0,frame_height*frame_width*sizeof(unsigned char));
    std::vector<Point2f> roivec;
    int counter=0;
    for(int i=0;i<length;i++)
    {
        Point2f p(aryPtr[i*2],aryPtr[i*2+1]);
        roivec.push_back(p);
    }
    for(int i=0;i<frame_height;i++)
    {
        for(int j=0;j<frame_width;j++)
        {
            if(pointPolygonTest(roivec,Point2f(j,i),true)>0)
            {
                h_segNeg[i*frame_width+j]=255;
                counter++;

            }
        }
    }
    std::cout<<counter<<std::endl;
    cudaMemcpy(d_segNeg,h_segNeg,frame_height*frame_width*sizeof(unsigned char),cudaMemcpyHostToDevice);
}
void KLTsparse_CUDA::findPoints()
{
    std::cout<<"applySegMask"<<std::endl;
    if(applyseg)
    {
        int nblocks = (frame_height*frame_width)/nFeatures;
        applySegMask<<<nblocks,nFeatures>>>(d_mask,d_segmask,d_segNeg);
    }
    std::cout<<"detector"<<std::endl;
    (*detector)(gpuGray, gpuCorners,maskMat);
    gpuCorners.download(corners);
}
void KLTsparse_CUDA::filterTrack()
{
    int addidx=0,lostcount=0;
    std::cout<<"for loop"<<std::endl;
    for (int k = 0; k < nFeatures; k++)
    {
        int statusflag = status.at<int>(k);
        Vec2f trkp = nextPts.at<Vec2f>(k);
        bool lost=false;
        if ( statusflag)
        {
            pttmp.x = trkp[0];
            pttmp.y = trkp[1];
            pttmp.t = frameidx;
            trackBuff[k].updateAFrame(&pttmp);
            if (!checkTrackMoving(trackBuff[k]))lost=true;
        }
        else
        {
            lost=true;
        }
        if(lost)
        {
            trackBuff[k].clear();
            label_final[k]=0;
            if(lostcount<corners.size[1])
            {
                Vec2f cnrp = corners.at<Vec2f>(lostcount++);
                pttmp.x = cnrp[0];
                pttmp.y = cnrp[1];
                pttmp.t = frameidx;
                trackBuff[k].updateAFrame(&pttmp);
                nextPts.at<Vec2f>(k)=cnrp;
            }
        }
        int x =trackBuff[k].cur_frame_ptr->x,
                y=trackBuff[k].cur_frame_ptr->y;
        if (trackBuff[k].len > minTrkLen)
        {

            ofvtmp.x0 = trackBuff[k].getPtr(trackBuff[k].len - minTrkLen-1)->x;
            ofvtmp.y0 = trackBuff[k].getPtr(trackBuff[k].len - minTrkLen-1)->y;
            ofvtmp.x1 = trackBuff[k].cur_frame_ptr->x;
            ofvtmp.y1 = trackBuff[k].cur_frame_ptr->y;
            ofvtmp.len = trackBuff[k].len;
            ofvtmp.idx = k;
            ofvBuff.updateAFrame(&ofvtmp);
            items.push_back(k);

        }
        h_curvec[addidx * 2] = x;
        h_curvec[addidx * 2 + 1] = y;
        addidx++;
        for(int i=0;i<maxgroupN;i++)
        {
            FeatBuff& polybuff = cvxPts[i];
            if(polybuff.len>2)
            {
                Mat polyMat(1,polybuff.len,CV_32FC2,polybuff.data);
                if(pointPolygonTest(polyMat,Point2f(x,y),false)>-1)
                {
                    h_label[k]=i;
                }
            }
        }
    }
    std::cout<<"applyPersToMask"<<std::endl;
    cudaMemcpy(d_curvec, h_curvec, nFeatures*2*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask,d_roimask,frame_height*frame_width*sizeof(unsigned char),cudaMemcpyDeviceToDevice);
    dim3 block(32, 32,1);
    applyPersToMask<<<addidx,block>>>(d_mask,d_curvec, d_persMap);
    gpuPrePts.upload(nextPts);
}
void KLTsparse_CUDA::PointTracking()
{
    std::cout<<"tracker"<<std::endl;
    tracker->sparse(gpuPreGray, gpuGray, gpuPrePts, gpuNextPts, gpuStatus);
    gpuStatus.download(status);
    gpuNextPts.download(nextPts);
}
void KLTsparse_CUDA::Render(unsigned char * framedata)
{
    /*
    if(!d_cvx||!d_cvxLen)
    {
        gpu_zalloc(d_cvx,nFeatures*nFeatures,sizeof(TrkPts));
        gpu_zalloc(d_cvxLen,nFeatures,sizeof(int));
    }
    cudaMemset(d_cvx,0,nFeatures*nFeatures*sizeof(TrkPts));
    cudaMemset(d_cvxLen,0,nFeatures*sizeof(int));
    for(int i=0;i<maxgroupN;i++)
    {
        cudaMemcpy(d_cvx,cvxInt[i].data,nFeatures*cvxInt[i].len*sizeof(TrkPts),cudaMemcpyHostToDevice);
        int len = cvxInt[i].len;
        cudaMemcpy(d_cvxLen+i,&(len),sizeof(int),cudaMemcpyHostToDevice);
    }
    */

    if(curTrkingIdx&&h_bbox[curTrkingIdx*4+2]>2,h_bbox[curTrkingIdx*4+3]>2)
    {
        int x=h_bbox[curTrkingIdx*4],y=h_bbox[curTrkingIdx*4+1],
        x1=h_bbox[curTrkingIdx*4+2]+h_bbox[curTrkingIdx*4],y1=h_bbox[curTrkingIdx*4+3]+h_bbox[curTrkingIdx*4+1];
    UperLowerBound(x,0,frame_width-1);
    UperLowerBound(x1,0,frame_width-1);
    UperLowerBound(y,0,frame_height-1);
    UperLowerBound(y1,0,frame_height-1);
    Mat tmpmat(zoomH,zoomW,CV_8UC3,h_zoomFrame);
    Mat framemat(frame_height,frame_width,CV_8UC3,framedata);
    cv::Rect rect(Point2i(x,y),Point2i(x1,y1));
    if(rect.area()>10)
    {
    cv::Mat patchMat = framemat(rect);
    resize(patchMat,tmpmat,Size(zoomH,zoomW));
    //imshow("patch",tmpmat);
    }
    }
    cudaMemcpy(d_clrvec,h_clrvec,nFeatures*3*sizeof(unsigned char),cudaMemcpyHostToDevice);
    cudaMemset(d_renderMask,0,frame_width*frame_height*3*sizeof(unsigned char));
    cudaMemcpy(d_label,label_final,nFeatures*sizeof(int),cudaMemcpyHostToDevice);

    renderGroup<<<nFeatures,nFeatures>>>(d_renderMask,d_curvec ,d_label,d_clrvec,d_persMap,d_cvx,d_cvxLen);
    cudaDeviceSynchronize();
    int nblocks = (frame_height*frame_width)/nFeatures;
    renderFrame<<<nblocks,nFeatures>>>(d_renderMask,rgbMat.data);
    cudaMemcpy(framedata,rgbMat.data,frame_height*frame_width*3*sizeof(unsigned char),cudaMemcpyDeviceToHost);
    //cudaMemcpy(framedata,d_cvx,frame_height*frame_width*3*sizeof(unsigned char),cudaMemcpyDeviceToHost);
}
int KLTsparse_CUDA::updateAframe(unsigned char* framedata, int fidx)
{
    frameidx=fidx;
    std::cout<<"frameidx:"<<frameidx<<std::endl;
    gpuGray.copyTo(gpuPreGray);
    std::cout<<"gpuPreGray"<<std::endl;
    Mat curframe(frame_height,frame_width,CV_8UC3,framedata);

    rgbMat.upload(curframe);

    gpu::cvtColor(rgbMat,gpuGray,CV_RGB2GRAY);
    PointTracking();
    findPoints();
    ofvBuff.clear();
    items.clear();
    filterTrack();

    /** Grouping  **/
    if(h_gcount[curTrkingIdx]<1)updateFlag=true;
    if(groupOnFlag&&ofvBuff.len>0)
    {

        unsigned char * curnbor = d_neighbor+offsetidx*nFeatures*nFeatures;
        float * curCos = d_cosine+offsetidx*nFeatures*nFeatures;
        float * curVelo = d_velo+offsetidx*nFeatures*nFeatures*2;
        cudaMemset(curnbor,0,nFeatures*nFeatures);
        cudaMemset(curCos,0,nFeatures*nFeatures*sizeof(float));
        cudaMemset(curVelo,0,nFeatures*nFeatures*sizeof(float)*2);
        cudaMemset(d_ofvec, 0, nFeatures* sizeof(ofv));
        cudaMemcpy(d_ofvec, ofvBuff.data, ofvBuff.len*sizeof(ofv), cudaMemcpyHostToDevice);
        cudaMemset(d_distmat,0,nFeatures*nFeatures*sizeof(float));
        searchNeighborMap <<<ofvBuff.len, ofvBuff.len >>>(curnbor,curCos,curVelo,d_ofvec,d_distmat,d_persMap, nFeatures);
        cudaMemcpy(h_curnbor,curnbor,nFeatures*nFeatures,cudaMemcpyDeviceToHost);
        cudaMemcpy(h_distmat,d_distmat,nFeatures*nFeatures*sizeof(float),cudaMemcpyDeviceToHost);

        knn();
        cudaMemcpy(curnbor,h_curnbor,nFeatures*nFeatures,cudaMemcpyHostToDevice);
        cudaMemset(d_neighborD,0,nFeatures*nFeatures);
        neighborD<<<nFeatures,nFeatures>>>(d_neighbor,d_cosine,d_velo,d_neighborD,offsetidx);
        memcpy(h_prelabel ,h_label,nFeatures*sizeof(int));
        pregroupN = groupN;
        cudaMemcpy(h_neighborD,d_neighborD,nFeatures*nFeatures,cudaMemcpyDeviceToHost);

        bfsearch();

        memset(h_overlap,0,nFeatures*nFeatures*sizeof(int));
        for(int i=0;i<nFeatures;i++)
        {
            int prelabel = h_prelabel[i],label = h_label[i];
            if(prelabel&&label)
                h_overlap[prelabel*nFeatures+label]++;
        }

        reGroup();

        for(int i = 0;i<nFeatures;i++)
        {
            if(h_label[i])
            {
                h_label[i]=idxmap[h_label[i]];
            }
        }
        memset(tmpn,0,nFeatures);
        for(int i=0;i<nFeatures;i++)
        {
            if(h_prelabel[i]==h_label[i])
            {
                tmpn[h_label[i]]++;
            }
        }
        for(int i=0;i<nFeatures;i++)
        {
            if(tmpn[i])
                h_gAge[i]++;
            else
                h_gAge[i]=0;
        }
        if(updateFlag)
        {
            memcpy(label_final,h_label,nFeatures*sizeof(int));
        }

        int maxidx=0;
        memset(h_gcount,0,nFeatures*sizeof(int));
        memset(h_com,0,nFeatures*2*sizeof(float));
        if(calPolyGon)
        {
            for(int i=0;i<=maxgroupN;i++)
            {
                setPts[i].clear();
                cvxPts[i].clear();
                cvxInt[i].clear();
            }
        }
        for(int i=0;i<nFeatures;i++)
        {
            int gidx = label_final[i];
            if(gidx)
            {
                h_gcount[gidx]++;
                if(calPolyGon)
                {
                    cvxPnttmp.x=h_curvec[i*2];
                    cvxPnttmp.y=h_curvec[i*2+1];
                    setPts[gidx].push_back(cvxPnttmp);
                }
                h_com[gidx*2]+=trackBuff[i].cur_frame_ptr->x;
                h_com[gidx*2+1]+=trackBuff[i].cur_frame_ptr->y;
                if(trackBuff[i].len>1)
                {
                    h_bbstats[gidx*4+2]+=trackBuff[i].cur_frame_ptr->x-trackBuff[i].getPtr(trackBuff[i].len-2)->x;
                    h_bbstats[gidx*4+3]+=trackBuff[i].cur_frame_ptr->y-trackBuff[i].getPtr(trackBuff[i].len-2)->y;
                }
                if(gidx>maxidx)maxidx=gidx;
            }
        }

        if(maxidx>maxgroupN)maxgroupN=maxidx+1;
        int maxAge=0,maxAgeIdx=0;
        for(int i=1;i<=maxgroupN;i++)
        {
            std::cout<<h_gAge[i]<<",";
            if(h_gcount[i]>0&&maxAge<h_gAge[i])
            {
                maxAge=h_gAge[i];
                maxAgeIdx=i;
            }
        }
        std::cout<<maxAgeIdx<<std::endl;
        for(int i=1;i<=maxgroupN;i++)
        {
            cvxInt[i].clear();
            if(h_gcount[i]>0)
            {
                HSVtoRGB(h_clrvec+i*3,h_clrvec+i*3+1,h_clrvec+i*3+2,i/(maxgroupN+0.01)*360,1,1);
                h_com[i*2]/=float(h_gcount[i]);
                h_com[i*2+1]/=float(h_gcount[i]);
                h_bbstats[i*4+2]=h_bbstats[i*4+2]/float(h_gcount[i]);
                h_bbstats[i*4+3]=h_bbstats[i*4+3]/float(h_gcount[i]);
                if(calPolyGon)
                {
                    convex_hull(setPts[i],cvxPts[i]);


                    FeatBuff& fbuff = cvxPts[i];
                    TrkPts ip;
                    for(int j=0;j<fbuff.len;j++)
                    {
                        ip.x =fbuff.getPtr(j)->x+0.5,ip.y =fbuff.getPtr(j)->y+0.5;
                        cvxInt[i].updateAFrame(&ip);
                    }
                }
            }
            else
            {
                h_gAge[i]=0;
            }
        }
        if(updateFlag)
        {
            bbTrkBUff.clear();
            if(maxgroupN>1)
            {
                //curTrkingIdx==maxAgeIdx;
                //h_gAge[maxAgeIdx]=0;

                for(int i=(curTrkingIdx+1)%maxgroupN;i!=curTrkingIdx;i=(i+1)%maxgroupN)
                {
                    float area = h_bbox[i*4+2]*h_bbox[i*4+3];
                    if(h_gcount[i]>5&&area>100)
                    {
                        curTrkingIdx=i;
                        break;
                    }
                }

            }
            memcpy(h_precom,h_com,nFeatures*2*sizeof(float));

            memset(h_bbstats,0,nFeatures*4*sizeof(float));
            for(int i=0;i<nFeatures;i++)
            {
                int gidx = label_final[i];
                if(gidx)
                {
                    float dx = abs(trackBuff[i].cur_frame_ptr->x-h_com[gidx*2]);
                    float dy = abs(trackBuff[i].cur_frame_ptr->y-h_com[gidx*2+1]);
                    h_bbstats[gidx*4] = (dx>h_bbstats[gidx*4])*dx+(dx<h_bbstats[gidx*4])*h_bbstats[gidx*4];
                    h_bbstats[gidx*4+1]=(dy>h_bbstats[gidx*4+1])*dy+(dy<h_bbstats[gidx*4+1])*h_bbstats[gidx*4+1];
//                    h_bbstats[gidx*4]+= dx*dx;
//                    h_bbstats[gidx*4+1]+=dy*dy;
                }
            }

            for(int i=1;i<=maxgroupN;i++)
            {
                if(h_gcount[i]>0)
                {
                    /*
                    h_bbstats[i*4] =sqrt(h_bbstats[i*4]/float(h_gcount[i])),
                    h_bbstats[i*4+1] = sqrt(h_bbstats[i*4+1]/float(h_gcount[i]));
                    if(h_bbstats[i*4]<h_bbstats[i*4+1])h_bbstats[i*4]=h_bbstats[i*4+1];
                    */
                    int indicator = h_com[i*2]<frame_width;
                    int x=indicator*h_com[i*2]+(!indicator)*(frame_width-1);
                    indicator = h_com[i*2+1]<frame_height;
                    int y=indicator*h_com[i*2+1]+(!indicator)*(frame_height-1);

                    float curLen = h_persMap[y*frame_width+x]/2;
                    if(h_bbstats[i*4]<curLen)h_bbstats[i*4]=curLen;
                    if(h_bbstats[i*4+1]<curLen)h_bbstats[i*4+1]=curLen;
                }
            }

        }

        pttmp.x=h_precom[curTrkingIdx*2];
        pttmp.y=h_precom[curTrkingIdx*2+1];
        bbTrkBUff.updateAFrame(&pttmp);
        for(int i=1;i<=maxgroupN;i++)
        {
            if(h_gcount[i]>0)
            {
                float devix = h_bbstats[i*4]*1.5,deviy = h_bbstats[i*4+1]*1.5;
                //devix=h_bbstats[i*4];
                //deviy=h_bbstats[i*4];
                h_precom[i*2]=h_bbstats[i*4+2]+h_precom[i*2];
                h_precom[i*2+1]=h_bbstats[i*4+3]+h_precom[i*2+1];
                h_bbox[i*4]=h_precom[i*2]-devix,
                h_bbox[i*4+1]=h_precom[i*2+1]-deviy,
                        h_bbox[i*4]=h_com[i*2]-devix,
                        h_bbox[i*4+1]=h_com[i*2+1]-deviy,
                h_bbox[i*4+2]=devix*2,
                h_bbox[i*4+3]=deviy*2;
                h_bbstats[i*4+2]=0;
                h_bbstats[i*4+3]=0;
            }
        }
        Render(framedata);
        offsetidx=(offsetidx+1)%TIMESPAN;
    }
    return 1;
}
void KLTsparse_CUDA::knn()
{
    for(int i=0;i<ofvBuff.len;i++)
    {
        int ridx = ofvBuff.getPtr(i)->idx;
        int maxidx=0;
        for(int k=0;k<KnnK;k++)
        {
            h_KnnIdx[k]=-1;
        }
        for(int j=0;j<nFeatures;j++)
        {
            if(h_curnbor[ridx*nFeatures+j])
            {
                float val = h_distmat[ridx*nFeatures+j];
                if(h_KnnIdx[maxidx]<0|| val<h_distmat[ridx*nFeatures+h_KnnIdx[maxidx]])
                {
                    h_KnnIdx[maxidx]=j;
                    int maxi=0;
                    for(int k=0;k<KnnK;k++)
                    {
                        if(h_KnnIdx[k]<0)
                        {
                            maxi=k;
                            break;
                        }
                        else if(h_distmat[ridx*nFeatures+h_KnnIdx[k]]>h_distmat[ridx*nFeatures+h_KnnIdx[maxi]])
                        {
                                maxi=k;
                        }
                    }
                    maxidx=maxi;
                }
            }
        }
        memset(h_curnbor+ridx*nFeatures,0,nFeatures);
        for(int k=0;k<KnnK;k++)
        {
            if(h_KnnIdx[k]>=0)
            {
                h_curnbor[ridx*nFeatures+h_KnnIdx[k]]=1;
            }
        }
    }
}
void KLTsparse_CUDA::reGroup()
{
    memset(idxmap,0,nFeatures*sizeof(int));
    memset(tmpn,0,nFeatures*sizeof(int));

    int vaccount=0;
    for(int i=1;i<=pregroupN;i++)
    {
        int maxcount=0,maxidx=0;
        for(int j=1;j<=groupN;j++)
        {
            if( h_overlap[i*nFeatures+j]>maxcount)
            {
                maxidx = j;
                maxcount = h_overlap[i*nFeatures+j];
            }
        }
        if(maxidx)
        {
            if(idxmap[maxidx])
            {
                if(h_overlap[i*nFeatures+maxidx]>h_overlap[idxmap[maxidx]*nFeatures+maxidx])
                {
                    tmpn[vaccount++]=idxmap[maxidx];
                    idxmap[maxidx]=i;
                }
                else
                {
                    tmpn[vaccount++]=i;
                }
            }
            else
                idxmap[maxidx]=i;
        }
        else
        {
            tmpn[vaccount++]=i;
        }
    }
    int vci=0;
    for(int i=1;i<=groupN;i++)
    {
        if(!idxmap[i])
        {
            if(vci<vaccount)
                idxmap[i]=tmpn[vci++];
            else
                idxmap[i]=(++pregroupN);
        }
    }
}
void KLTsparse_CUDA::bfsearch()
{
    int pos=0;
    bool isempty=false;
    int gcount=0;
    curK=1;
    groupN=0;
    memset(idxmap,0,nFeatures*sizeof(int));
    memset(tmpn,0,nFeatures*sizeof(int));
    memset(h_label,0,nFeatures*sizeof(int));
    memset(h_gcount,0,nFeatures*sizeof(int));
    int idx = items[pos];
    h_label[idx]=curK;
    for(int i=0;i<nFeatures;i++)
    {
        tmpn[i]=(h_neighborD[idx*nFeatures+i]);
    }
    items[pos]=0;
    gcount++;
    while (!isempty) {
        isempty=true;
        int ii=0;
        for(pos=0;pos<items.size();pos++)
        {
            idx=items[pos];
            if(idx)
            {
                if(ii==0)ii=pos;
                isempty=false;
                if(tmpn[idx])
                {
                    int nc=0,nnc=0;
                    for(int i=0;i<nFeatures;i++)
                    {
                        if(h_neighborD[idx*nFeatures+i])
                        {
                            nc++;
                            //if(tmpn[i])nnc++;
                            nnc+=(tmpn[i]>0);
                        }
                    }
                    if(nnc>nc*coNBThresh)
                    {
                        gcount++;
                        h_label[idx]=curK;
                        for(int i=0;i<nFeatures;i++)
                        {
                            tmpn[i]+=h_neighborD[idx*nFeatures+i];
                        }
                        items[pos]=0;
                        if(ii==pos)ii=0;
                    }
                }
            }
        }
        if(gcount>0)
        {
            h_gcount[curK]+=gcount;
            gcount=0;
        }
        else if(!isempty)
        {
            if(h_gcount[curK]>minGSize)
            {
                groupN++;
                idxmap[curK]=groupN;
            }
            curK++;
            gcount=0;
            memset(tmpn,0,nFeatures*sizeof(int));
            pos=ii;
            idx=items[pos];
            gcount++;
            h_label[idx]=curK;
            for(int i=0;i<nFeatures;i++)
            {
                tmpn[i]+=h_neighborD[idx*nFeatures+i];
            }
            items[pos]=0;
        }
    }
    for(int i=0;i<nFeatures;i++)
    {
        h_label[i]=idxmap[h_label[i]];
    }

}
