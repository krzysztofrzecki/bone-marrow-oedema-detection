#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <queue>
#include <vector>
#include <algorithm>
#include "CImg.h"

#include "CImg.h"

using namespace cimg_library;
using namespace std;

class Punkt
{
public:
  int x;
  int y;
  int z;
  double kolor;

  Punkt(){}
  Punkt(int X,int Y,double C) {x=X;y=Y;kolor=C;}
  Punkt(int X,int Y) {x=X;y=Y;}
  Punkt(int X,int Y,int Z,double C) {x=X;y=Y;z=Z;kolor=C;}
  Punkt(int X,int Y,int Z) {x=X;y=Y;z=Z;}
};

bool operator< (const Punkt& p1, const Punkt &p2)
{
	return p1.kolor > p2.kolor;	
}

bool operator> (const Punkt& p1, const Punkt &p2)
{
	return p1.kolor < p2.kolor;	
}

bool PointCompFunction (Punkt P1,Punkt P2) { return (P1.kolor < P2.kolor); }

vector<Punkt> FindShortestPath(CImg<unsigned char> &dane,int label1, int label2,double MAX_TH);
void FindJointLine(vector<Punkt> &jointLine,double EPS);
void FindJointRegion(vector<Punkt> &jointLine,CImg<unsigned char> &dane,int label1, int label2,double JOINT);
void FindQuadrants(vector<Punkt> &jointLine,CImg<unsigned char> &dane,int label1, int label2);
void FindReference(CImg<unsigned char> &labels,int label,double TH);



#define LEFT_BONE 1
#define RIGHT_BONE 3
#define MIDDLE_BONE 2
#define BACKGROUND 0
#define ROI_MARKER 10

#define REFERENCE 50
#define LEFT_BONE_TOP 60
#define LEFT_BONE_BOTTOM 70
#define RIGHT_BONE_TOP 80
#define RIGHT_BONE_BOTTOM 90
#define LEFT_MIDDLE_BONE_TOP 100
#define LEFT_MIDDLE_BONE_BOTTOM 110
#define RIGHT_MIDDLE_BONE_TOP 120
#define RIGHT_MIDDLE_BONE_BOTTOM 130


double MAX_THICKNESS_FOR_PRESENCE_OF_JOINT = 15;
double TOLERANCE = 0.2;
double JOINT_REGION_THICKNESS = 17.;
double LENGTH_THRESHOLD_1 = 17.;
double LENGTH_THRESHOLD_2 = 50.;

double CENTER_LINE;

double REFERENCE_REGION_THICNKESS = 10.;



int main(int argc,char *argv[])
{

    int num = atoi(argv[2]);
	int width = atoi(argv[3]);
	int height = atoi(argv[4]);
	int depth = atoi(argv[5]);

    int read = 0;

    if (argc==7) read = atoi(argv[6]);

    cerr << read << endl;

    if (read)
    {
    cerr << endl << endl << "Dane" << endl << endl;
        ifstream in;
        in.open("dane");
            in >> MAX_THICKNESS_FOR_PRESENCE_OF_JOINT;
            in >> JOINT_REGION_THICKNESS;
            in >> LENGTH_THRESHOLD_1 ;
            in >> LENGTH_THRESHOLD_2;
            in >> TOLERANCE;
        in.close();
    }

    if (width!=400)
    {
        MAX_THICKNESS_FOR_PRESENCE_OF_JOINT = MAX_THICKNESS_FOR_PRESENCE_OF_JOINT*width/400.;
        JOINT_REGION_THICKNESS = JOINT_REGION_THICKNESS*(double)width/400.;
        LENGTH_THRESHOLD_1 = LENGTH_THRESHOLD_1*(double)width/400.;
        LENGTH_THRESHOLD_2 = LENGTH_THRESHOLD_2*(double)width/400.;
        REFERENCE_REGION_THICNKESS = REFERENCE_REGION_THICNKESS*(double)width/400.;
    }


	CImg<unsigned char> labels(width,height,depth);

    char name[1000];
    sprintf(name,"%s_%d_%d_%d_%d_1_.raw",argv[1],num,width,height,depth);

	labels.load_raw(name,width,height,depth);

	labels.display();

	FindReference(labels,MIDDLE_BONE,REFERENCE_REGION_THICNKESS);

	for(int k=0;k<depth;k++)
	{

		CImg<unsigned char> dane(width,height);
		for(int i=0;i<width;i++)
		for(int j=0;j<height;j++)
		{
			dane(i,j) = labels(i,j,k);
		}

// Left side
		vector<Punkt> jointLM = FindShortestPath(dane,LEFT_BONE,MIDDLE_BONE,MAX_THICKNESS_FOR_PRESENCE_OF_JOINT);

        cerr << "L";
		if (jointLM.size())
		{
			FindJointLine(jointLM,TOLERANCE);
        cerr << "L";
			FindJointRegion(jointLM,dane,LEFT_BONE,MIDDLE_BONE,JOINT_REGION_THICKNESS);
        cerr << "L";
			FindQuadrants(jointLM,dane,LEFT_BONE,MIDDLE_BONE);
        cerr << "L";
		}
        cerr << endl;

// Right side
		vector<Punkt> jointRM = FindShortestPath(dane,RIGHT_BONE,MIDDLE_BONE,MAX_THICKNESS_FOR_PRESENCE_OF_JOINT);
        cerr << "R";
		if (jointRM.size()) 
		{
			FindJointLine(jointRM,TOLERANCE);
        cerr << "R";
			FindJointRegion(jointRM,dane,RIGHT_BONE,MIDDLE_BONE,JOINT_REGION_THICKNESS);
        cerr << "R";
			FindQuadrants(jointRM,dane,RIGHT_BONE,MIDDLE_BONE);
        cerr << "R";
		}
        cerr << endl;

		for(int i=0;i<jointLM.size();i++)
			dane(jointLM[i].x,jointLM[i].y) = 200;
		for(int i=0;i<jointRM.size();i++)
			dane(jointRM[i].x,jointRM[i].y) = 200;

		for(int i=0;i<width;i++)
		for(int j=0;j<height;j++)
		{
			labels(i,j,k) = dane(i,j);
			if (labels(i,j,k) > BACKGROUND && labels(i,j,k)<REFERENCE) labels(i,j,k) = 255;
		}

	}



    sprintf(name,"Regions_%d_%d_%d_%d_1_.raw",num,width,height,depth);
	labels.save_raw(name);

    labels.display("labels");
	
	
}


/***********************************************************************************/
/***********************************************************************************/
/***********************************************************************************/

void FindQuadrants(vector<Punkt> &jointLine,CImg<unsigned char> &dane,int label1, int label2)
{
	double d = sqrt((jointLine[0].x-jointLine[jointLine.size()-1].x)*(jointLine[0].x-jointLine[jointLine.size()-1].x)
				+(jointLine[0].y-jointLine[jointLine.size()-1].y)*(jointLine[0].y-jointLine[jointLine.size()-1].y));

//	cerr << "Start (" << jointLine[0].x << ", " << jointLine[0].y << "), end (" << jointLine[jointLine.size()-1].x << ", " << jointLine[jointLine.size()-1].y << ")" << " Distance=" << d << endl;

	if (d < LENGTH_THRESHOLD_1) 
	{
//		cerr << "Case 1" << endl;
		return;			//No quadrants
	}

	if (d < LENGTH_THRESHOLD_2)					//Only top quadrants
	{
//		cerr << "Case 2" << endl;
		for(int i=0;i<dane.width();i++)
		for(int j=0;j<dane.height();j++)
		{
			if (dane(i,j) == label1 + ROI_MARKER )
			{
				if (label1 == LEFT_BONE)
					dane(i,j) = LEFT_BONE_TOP;
				else
					dane(i,j) = RIGHT_BONE_TOP;
			}
			else if (dane(i,j) == label2 + ROI_MARKER)
			{
				dane(i,j) = RIGHT_MIDDLE_BONE_TOP;
			}
			else if (dane(i,j) == label2 + 2*ROI_MARKER)
			{
				dane(i,j) = LEFT_MIDDLE_BONE_TOP;
			}
		}
	}
	else
	{
//		cerr << "Case 3" << endl;
		double xc = jointLine[jointLine.size()/2].x;
		double yc = jointLine[jointLine.size()/2].y;

		double mx = 0.;
		double my = 0.;
		double sxy = 0.;
		double sxx = 0.;
		double a,b;
		int PADDING = jointLine.size()/4;
		for(int i=jointLine.size()/2-PADDING;i<=jointLine.size()/2 + PADDING;i++)
		{
			mx += jointLine[i].y;
			my += jointLine[i].x;
			sxx += jointLine[i].y*jointLine[i].y;
			sxy += jointLine[i].x*jointLine[i].y;
		}
		mx /= (double)(2.0*PADDING+1.0);
		my /= (double)(2.0*PADDING+1.0);
		sxx /= (double)(2.0*PADDING+1.0);
		sxy /= (double)(2.0*PADDING+1.0);
		a = (sxy - mx*my)/(sxx - mx*mx);
		a = -1./a;
		b = xc - a * yc;
		for(int i=0;i<dane.width();i++)
		for(int j=0;j<dane.height();j++)
		{
			if (dane(i,j) == label1 + ROI_MARKER )
			{
				if (label1 == LEFT_BONE)
				{
					if (a*j+b > i)
						dane(i,j) = LEFT_BONE_TOP;
					else
						dane(i,j) = LEFT_BONE_BOTTOM;
				}
				else
				{
					if (a*j+b < i)
						dane(i,j) = RIGHT_BONE_TOP;
					else
						dane(i,j) = RIGHT_BONE_BOTTOM;
				}
			}
			else if (dane(i,j) == label2 + ROI_MARKER)
			{
				if (a*j+b < i)
					dane(i,j) = RIGHT_MIDDLE_BONE_TOP;
				else
					dane(i,j) = RIGHT_MIDDLE_BONE_BOTTOM;
			}
			else if (dane(i,j) == label2 + 2*ROI_MARKER)
			{
				if (a*j+b > i)
					dane(i,j) = LEFT_MIDDLE_BONE_TOP;
				else
					dane(i,j) = LEFT_MIDDLE_BONE_BOTTOM;
			}
		}
	}
}

/***********************************************************************************/
/***********************************************************************************/
/***********************************************************************************/

void FindReference(CImg<unsigned char> &labels,int label,double TH)
{

	int maxLicz = 0;
	int kMax = 0;
	for(int k=0;k<labels.depth();k++)
	{
		int licz = 0;
		for(int i=0;i<labels.width();i++)
		for(int j=0;j<labels.height();j++)
		{
			if (labels(i,j,k) == label) licz++;
		}
		if (licz > maxLicz)
		{
			maxLicz = licz;
			kMax = k;
		}
	}

	CImg<unsigned char> dum2D(labels.width(),labels.height());
	dum2D.fill(0);

	queue<Punkt> kolejka;

	for(int i=0;i<labels.width();i++)
	{
		if (labels(i,0,kMax) != label) 
		{
			kolejka.push(Punkt(i,0));
			dum2D(i,0) = 1;
		}
		if (labels(i,labels.height()-1,kMax) != label) 
		{
			kolejka.push(Punkt(i,labels.height()-1));
			dum2D(i,labels.height()-1) = 1;
		}
	}

	while(kolejka.size())
	{
		Punkt punkt = kolejka.front();
		kolejka.pop();
		int I = punkt.x;
		int J = punkt.y;
		for(int i1=-1;i1<=1;i1++)
		for(int j1=-1;j1<=1;j1++)
		{
			if (abs(i1)+abs(j1)==1 && I+i1>=0 && J+j1>=0 && I+i1<labels.width() && J+j1<labels.height())
			{
				if (labels(I+i1,J+j1,kMax)!=label && dum2D(I+i1,J+j1) == 0)
				{
					punkt.x = I+i1;
					punkt.y = J+j1;
					dum2D(I+i1,J+j1) = 1;
					kolejka.push(punkt);
				}
			}
		}
	}

	double xx,yy,xy,xc,yc,n;

	xx = 0.;
	yy = 0.;
	xy = 0.;
	xc = 0.;
	yc = 0.;
	n = 0;

	for(int i=0;i<dum2D.width();i++)
	for(int j=0;j<dum2D.height();j++)
	{
		if (dum2D(i,j)==0)
		{
			xc += i;
			yc += j;
			n += 1.;
		}
	}
	xc /= n;
	yc /= n;

	CENTER_LINE = xc;

	for(int i=0;i<dum2D.width();i++)
	for(int j=0;j<dum2D.height();j++)
	{
		if (dum2D(i,j)==0)
		{
			xx += (i-xc)*(i-xc);
			yy += (j-yc)*(j-yc);
			xy += (i-xc)*(j-yc);
		}
	}

	double l1 = 0.5*(xx+yy-sqrt((xx-yy)*(xx-yy)+4.*xy*xy));
	double l2 = 0.5*(xx+yy+sqrt((xx-yy)*(xx-yy)+4.*xy*xy));

	double a = xx - l2;
	double b =xy;

	double x,y;

	if (a>b)
	{
		y = 1.0;
		x = -b/a;
	}
	else
	{
		x = 1.0;
		y = - a/b;
	}

	unsigned char l[3] = {label+1,label+1,label+1};
	double dy = yc;
//	double dx = -dy*atan2(x,y);
	double dx = 0.;
	dum2D.draw_line(xc,yc,xc - dx,yc-dy,l);

	dy =  yc - (dum2D.height()-1);
//	dx = -dy*atan2(x,y);
	dum2D.draw_line(xc,yc,xc - dx,yc-dy,l);

	for(int k=0;k<labels.depth();k++)
	{
		CImg<float> dist(labels.width(),labels.height());
		CImg<unsigned char> dum(labels.width(),labels.height());
		dum.fill(0);
		int flag = 0;
		for(int i=0;i<dum2D.width();i++)
		for(int j=0;j<dum2D.height();j++)
		{
			if (dum2D(i,j) == label+1 && labels(i,j,k)==label)
			{
				dum(i,j) = REFERENCE;
				flag = 1;
			}
		}
		if (flag == 0) continue;
		dist = dum.get_distance(REFERENCE);
		for(int i=0;i<dum2D.width();i++)
		for(int j=0;j<dum2D.height();j++)
		{
			if (dist(i,j) <=TH && labels(i,j,k)==label)
			{
				dum(i,j) = REFERENCE;
			}
		}
		for(int i=0;i<dum2D.width();i++)
		for(int j=0;j<dum2D.height();j++)
		{
			if (dum(i,j)==REFERENCE) labels(i,j,k) = REFERENCE;
		}
	}

}

/***********************************************************************************/
/***********************************************************************************/
/***********************************************************************************/

void FindJointRegion(vector<Punkt> &jointLine,CImg<unsigned char> &dane,int label1, int label2,double JOINT)
{

	CImg<float> distance(dane.width(),dane.height());

	distance = dane.get_distance(label1);
	double dmax = 0;
	for(int i=0;i<jointLine.size();i++)
	{
		if (distance(jointLine[i].x,jointLine[i].y) > dmax) dmax = distance(jointLine[i].x,jointLine[i].y);
	}
	
	double TH = dmax + 1.0;
	
	CImg<unsigned char> dum(dane.width(),dane.height());
	dum.fill(0);
	for(int i=0;i<jointLine.size();i++)
		dum(jointLine[i].x,jointLine[i].y) = 255;

	distance = dum.get_distance(255);

	for(int i=0;i<dane.width();i++)
	for(int j=0;j<dane.height();j++)
	{
		if (dane(i,j) == label1 && distance(i,j)<=TH)
		{
			int flag = 0;
			for(int i1=-1;i1<=1;i1++)
			for(int j1=-1;j1<=1;j1++)
			{
				if (i+i1>=0 && i+i1<dane.width() && j+j1>=0 && j+j1<=dane.height())
				{
					if (dane(i+i1,j+j1) == BACKGROUND) flag = 1;
				}
			}
			if (flag==1) dum(i,j) = label1;
		}
	}

	for(int i=0;i<dane.width();i++)
	for(int j=0;j<dane.height();j++)
	{
		if (dane(i,j) == label2 && distance(i,j)<=TH)
		{
			int flag = 0;
			for(int i1=-1;i1<=1;i1++)
			for(int j1=-1;j1<=1;j1++)
			{
				if (i+i1>=0 && i+i1<dane.width() && j+j1>=0 && j+j1<=dane.height())
				{
					if (dane(i+i1,j+j1) == BACKGROUND) flag = 1;
				}
			}
			if (flag==1) dum(i,j) = label2;
		}
	}

	distance = dum.get_distance(label1);
	for(int i=0;i<dane.width();i++)
	for(int j=0;j<dane.height();j++)
	{
		if (dane(i,j) == label1 && distance(i,j)<=JOINT)
		{
			dane(i,j) = label1 + ROI_MARKER;
		}
	}

	distance = dum.get_distance(label2);
	for(int i=0;i<dane.width();i++)
	for(int j=0;j<dane.height();j++)
	{
		if (dane(i,j) == label2 && distance(i,j)<=JOINT)
		{
			if (label1 == RIGHT_BONE)
				dane(i,j) = label2 + ROI_MARKER;
			else
				dane(i,j) = label2 + 2*ROI_MARKER;
		}
	}

}

/***********************************************************************************/
/***********************************************************************************/
/***********************************************************************************/

#define MAX_PADDING 20
#define SPLIT_FACTOR 5

void FindJointLine(vector<Punkt> &jointLine,double TOL)
{

	int PADDING = jointLine.size()/SPLIT_FACTOR;
	if (PADDING > 20) PADDING = MAX_PADDING;

	vector<float> coefs;
	for(int splitPoint1 = PADDING/2;splitPoint1 < jointLine.size() - PADDING/2;splitPoint1++)
	{
		double mx1 = 0.;
		double my1 = 0.;
		double sxy1 = 0.;
		double sxx1 = 0.;
		double a1;

		for(int i=splitPoint1-PADDING/2;i<splitPoint1 + PADDING/2;i++)
		{
			mx1 += i;
			my1 += jointLine[i].kolor;
			sxx1 += i*i;
			sxy1 += i*jointLine[i].kolor;
		}
		mx1 /= (double)PADDING;
		my1 /= (double)PADDING;
		sxx1 /= (double)PADDING;
		sxy1 /= (double)PADDING;
		a1 = (sxy1 - mx1*my1)/(sxx1 - mx1*mx1);
		coefs.push_back(a1);
	}

	int first = 0;
	int last = coefs.size()-1;

	for(first=0;first<coefs.size();first++)
		if (coefs[first]<TOL) break;

	for(last=coefs.size()-1;last>=0;last--)
		if (coefs[last]<TOL) break;

	jointLine.assign(jointLine.begin()+first+PADDING/2,jointLine.begin()+last+PADDING/2);

}

/***********************************************************************************/
/***********************************************************************************/
/***********************************************************************************/

std::vector<Punkt> FindShortestPath(CImg<unsigned char> &dane,int label1, int label2,double MAX_TH)
{

	int licz1=0,licz2=0;
	std::vector<Punkt> lista;
	lista.clear();


	for(int i=0;i<dane.width();i++)
	for(int j=0;j<dane.height();j++)
	{
		if (dane(i,j)==label1) licz1++;
		if (dane(i,j)==label2) licz2++;
	}
	if (licz1*licz2==0) return lista;

	CImg<float> distance1(dane.width(),dane.height());
	CImg<float> distance2(dane.width(),dane.height());

	distance1 = dane.get_distance(label1);
	distance2 = dane.get_distance(label2);

	for(int i=0;i<distance1.width();i++)
	for(int j=0;j<distance1.height();j++)
		distance1(i,j) = fabs(distance1(i,j) - distance2(i,j));

	vector<Punkt> S;
	for(int i=0;i<distance1.width();i++)
	{
		S.push_back(Punkt(i,0,1));
	}
	

	CImg<signed char> way(dane.width(),dane.height());
	CImg<float> dist(dane.width(),dane.height());
	
	dist.fill(ULONG_MAX);
	way.fill(-1);
	
	for(int i=0;i<S.size();i++)
	{
		dist(S[i].x,S[i].y) = distance1(S[i].x,S[i].y);
		way(S[i].x,S[i].y) = 0;
	}

	priority_queue<Punkt,std::vector<Punkt>,less<Punkt> > kolejka;

	for(int i=0;i<S.size();i++)
	{
		S[i].kolor = 0;
		kolejka.push(S[i]);
	}

	Punkt punkt;
	
	int I=0,J=0;
	while(kolejka.size())
	{
		punkt = kolejka.top();
		kolejka.pop();
		I = punkt.x;
		J = punkt.y;
		if (J==dane.height()-1) break;
		for(int i1=-1;i1<=1;i1++)
		for(int j1=-1;j1<=1;j1++)
		{
			if (abs(i1)+abs(j1)==1 && I+i1>=0 && J+j1>=0 && I+i1<dane.width() && J+j1<dane.height())
			{
				if (dist(I+i1,J+j1)>distance1(I+i1,J+j1)+dist(I,J))
				{
					punkt.x = I+i1;
					punkt.y = J+j1;
					dist(I+i1,J+j1) = distance1(I+i1,J+j1) + dist(I,J);
					punkt.kolor = dist(I+i1,J+j1);
					kolejka.push(punkt);
					way(I+i1,J+j1) = 10*i1 + j1;
				}
			}
		}
	}

	distance1 = dane.get_distance(label1);
	double dmin = 1e30;
	for(int i=0;i<dane.width();i++)
	for(int j=0;j<dane.height();j++)
	{
		if (dane(i,j) == label2)
		{
			if (distance1(i,j) < dmin) dmin = distance1(i,j);
		}
	}

	if (dmin > MAX_TH) return lista;		//jeśli odległość label1 od label2 większa od ustalonego progu - nie ma stawu

	lista.push_back(Punkt(I,J,distance1(I,J)));

	while(way(I,J)!=0)
	{
		if (way(I,J)==10) I = I-1;
		else if (way(I,J)==-10) I = I+1;
		else if (way(I,J)==1) J = J-1;
		else J = J+1;
		lista.push_back(Punkt(I,J,distance1(I,J)));		//distance1(I,J) to połowa odległości między label1 i label2 w lokalizacji (I,J)
	}

	int first = 0;
	int last = lista.size()-1;

	for(first=0;first<lista.size();first++)
		if (lista[first].kolor<MAX_TH) break;

	for(last=lista.size()-1;last>=0;last--)
		if (lista[last].kolor<MAX_TH) break;

	lista.assign(lista.begin()+first,lista.begin()+last);

	return lista;

}



