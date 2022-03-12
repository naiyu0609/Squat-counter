#include "opencv2/opencv.hpp"
#include <iostream>
#include <time.h>
#define FRAME_TOTAL 1000 //�`�@Ū��Frame��

using namespace cv; //�ŧi opencv �禡�w���R�W�Ŷ�
using namespace std; //�ŧi C++�禡�w���R�W�Ŷ�

/** Function Headers */
void detectAndDisplay(Mat frame); //�����H�y�P��ܪ��禡�ŧi

/** Global variables */
String face_cascade_name = "data\\haarcascade_frontalface_alt.xml";//�����H�y�r�������V�m�ƾ�
//��M�פU�G"lbpcascade_frontalface.xml��(��sln�ɩ�@�_)
//�۹���|�G"data/lbpcascade_frontalface.xml��(��b�U�@�h��data�ɮק�)
//������|�G"D: / AAA / BBB / CCC / lbpcascade_frontalface.xml��(���N��m)

CascadeClassifier face_cascade; //�إ��r������������

Mat frame, face_im, face_ROI;
int cross=1; //�O�_��V�o���u
int frame_no = 0; //�e���s��
int counter = 0; //�����i���ϥ�
int times = 0; //�o��
int outside=0; //�O�_�b�o���Ϥ�
int face_check = 0; //���L�������y
string s; //�o���r��
int y_trajectory[FRAME_TOTAL]; //������V�y��
Rect buffer[4]; //�e�|�ӵe�����H�yROI��m(�Ȧs�A�Ω�ROI�y�񥭷Ƥ�)
vector<Rect> faces;//�s�y
clock_t start, End; // �x�s�ɶ��Ϊ��ܼ�


int main() {
    cout << "/* �`�ۭp�ƾ� */" << endl;
    VideoCapture capture;//Ū���۾�

    //-- 1. Load the cascades
    if (!face_cascade.load(face_cascade_name)) { printf("--(!)Error loading face cascade\n"); return -1; };

    //-- 2. Read the video stream
    capture.open(0);
    if (!capture.isOpened()) { printf("--(!)Error opening video capture\n"); return -1; }

    start = clock();//�����}�l�ɶ�

    while (frame_no< FRAME_TOTAL)//Frame�ƩT�w
    {
        capture.read(frame);//ŪFrame

        if (frame.empty())//�p�G�SŪ��
        {
            printf(" --(!) No captured frame -- Break!");
            break;
        }
        
        //-- 3. Apply the classifier to the frame
        detectAndDisplay(frame);

        frame_no++;//Frame��+1
        counter++;//�i����+1

        int c = waitKey(10);//Ū��L
        if ((char)c == 27) {//�p�G��ESC
            cout << "Break!" << endl;
            break; } // escape
    }
    End = clock();//�����ɶ�

    cout << "�`�@�ۤU" << times << "���@"<< round(double(End - start) / CLK_TCK) <<"��" << endl;//�L�X���G
    waitKey();//���ݫ���L
    return 0;
}

//�����H�y�P��ܨ禡
void detectAndDisplay(Mat frame)
{
    /*�H�y��������*/
    Mat im_gray; //�Ƕ��v������
    cvtColor(frame, im_gray, COLOR_BGR2GRAY);//�m��v����Ƕ�
    equalizeHist(im_gray, im_gray);//�Ƕ��Ȥ�ϵ���(���۰ʼW�j)

    //-- Detect faces
    face_cascade.detectMultiScale(im_gray, faces, 1.1, 3, 0, Size(80, 80));
    
    //���H�y���B�z
    if (faces.size() > 0) {
        face_check = 1;//���������y
        face_ROI = frame(Rect(faces[0].x, faces[0].y, faces[0].width, faces[0].height));//�N�H�y�����sROI
        face_ROI.convertTo(face_im, face_im.type());//�N�H�y�v���s���ܼ�

        //�N�H�yroi�ƻs��buffer[0]
        buffer[0].x = faces[0].x;
        buffer[0].y = faces[0].y;
        buffer[0].width = faces[0].width;
        buffer[0].height = faces[0].height;

        //�Nbuffer[0]��[3]�� x, y, width, height �|���ؤ��O�����A���Nfaces[0]�A�ت��O��֤H�y�x�ήت��ݰ�
        for (int i = 1; i < 4; i++) {
            faces[0].x += buffer[i].x;
            faces[0].y += buffer[i].y;
            faces[0].width += buffer[i].width;
            faces[0].height += buffer[i].height;
        }
        faces[0].x /= 4;
        faces[0].y /= 4;
        faces[0].width /= 4;
        faces[0].height /= 4;

        //�h��buffer�����: ��buffer[1]��[3]���O�x�s[0]��[2]����ơA�]�N�O�̪�T�ӵe�����H�yroi
        for (int i = 3; i >0; i--) {
            buffer[i].x = buffer[i-1].x;
            buffer[i].y = buffer[i-1].y;
            buffer[i].width = buffer[i-1].width;
            buffer[i].height = buffer[i-1].height;
        }
        /*�p������*/
        //�p��H�y�����I
        //�ھ� ROI �p��H�y����
        Point leftup(faces[0].x, faces[0].y);
        Point rightdown(faces[0].x + faces[0].width, faces[0].y + faces[0].height);
        Point center(faces[0].x + faces[0].width / 2, faces[0].y + faces[0].height / 2);

        //�by_trajectory������V�y��}�C�A�O���o�ӵe�����H�y������V�����I
        y_trajectory[frame_no] = center.y/2;

        if (center.x > 200 && center.x < frame.size().width - 200 && center.y > 50)//�o���ϰ�P�w
        {
            outside = 1;//�L�V��
            if (cross == 0 && center.y < frame.size[0] / 2) {//�p�w��U�ɤS�^�W��
                cross = 1;
                times++;//�o��
                cout << times << endl;
            }
            else if (cross == 1 && center.y > frame.size[0]/2) //�p�w��W�ɤS�^�U��
                cross = 0;
        }
        else if(outside==1)//�p�쥻�b�o���ϫ�W�X�o����
        {
            outside = 0;
            cout  << "�D�o���Ϥ�"<<endl;
        }

        /*ø�ϳ���*/
        //�ҽk�I����
        blur(frame, frame, Size(7, 7));
        //�N�M�����H�y�ƻs��ҽk�I���Ϫ��H�y��m
        face_im.copyTo(frame(Rect(buffer[1].x, buffer[1].y, buffer[1].width, buffer[1].height)));
        //ø�s�H�y�x�ή�
        cv::rectangle(frame, leftup, rightdown, Scalar(255, 0, 0));
        //�N�H�y�ƻs��ҽk�I���Ϫ��k�W��
        face_im.copyTo(frame(Rect(frame.size[1] - 5 - buffer[1].width, 25, buffer[1].width, buffer[1].height)));
    }
    else if(face_check==1){//�p�G�������y��S����
        face_check = 0;
        times = 0;
        start = clock();
        cout << "�y����" << endl;
    }

    putText(frame, "B10607044", Point(frame.size[1]/2 - 100, frame.size[0]-15), FONT_HERSHEY_COMPLEX | FONT_ITALIC, 1, CV_RGB(255, 0, 255), 1, LINE_AA);//�g�Ǹ�
    s = to_string(times);
    putText(frame, s, Point(20, 30), FONT_HERSHEY_COMPLEX | FONT_ITALIC, 1, CV_RGB(255, 0, 255), 1, LINE_AA);//�g����

    //-- Show what you got
    imshow("Squat detection", frame);
    moveWindow("Squat detection", 800, 0);

    Mat trajectory(Size(FRAME_TOTAL,frame.size[0]/2), frame.type(), Scalar(0));//�ж¼v��
    line(trajectory, Point(0, trajectory.size[0]/2), Point(trajectory.size[1], trajectory.size[0]/2), CV_RGB(255, 0, 0),2);//���ɽu
    for (int i = 1; i < counter; i++) {//�e�i��
        line(trajectory, Point(i, y_trajectory[i-1]), Point(i, y_trajectory[i]), CV_RGB(0, 255, 255), 4);
    }
    imshow("trajectory", trajectory);
    moveWindow("trajectory", 0, 500);

}