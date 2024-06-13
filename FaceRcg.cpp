#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>

using namespace std;
using namespace cv;

void DisplayOverlayFrame(Mat frame);
CascadeClassifier face_cascade;
CascadeClassifier nose_cascade;
String face_cascade_name;
String nose_cascade_name;


int main(int argc, const char** argv)
{
    CommandLineParser parser(argc, argv, "{camera|0|Camera device number.}");

    // Load the cascades    
    face_cascade_name = "C:\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_alt2.xml";
    // face_cascade_name = "haarcascade_frontalface_alt2.xml";
    if (!face_cascade.load(face_cascade_name))
    {
        cout << "--(!)Error loading face cascade\n";
        return -1;
    };

    nose_cascade_name = "C:\\opencv\\sources\\data\\haarcascades\\haarcascade_mcs_nose.xml";
    // nose_cascade_name = "haarcascade_mcs_nose.xml";
    if (!nose_cascade.load(nose_cascade_name))
    {
        cout << "--(!)Error loading face cascade\n";
        return -1;
    };

    // Read the video stream
    int camera_device = parser.get<int>("camera");
    VideoCapture video;
    video.open(camera_device);
    if (!video.isOpened())
    {
        cout << "--(!)Error opening video stream\n";
        return -1;
    }
	
    Mat frame;    
    // Capture frame each time
    while (video.read(frame))
    {
        if (frame.empty())
        {
            cout << "--(!) No captured frame -- Break!\n";
            break;
        }

	DisplayOverlayFrame(frame);

        if (waitKey(1) == 27)        
            break; // escapes        
    }
    return 0;
}

void DisplayOverlayFrame(Mat frame)
{
    Mat frame_gray;
    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);

    //-- Detect faces
    vector<Rect> faces;
    face_cascade.detectMultiScale(frame_gray, faces);

    // Load image for overlay: mustache.png
    Mat imgMustache = imread("mustache.png", IMREAD_UNCHANGED);

    // Create the mask for the mustache
    Mat orig_mask = imgMustache(Rect(0, 0, imgMustache.cols, imgMustache.rows));
    extractChannel(orig_mask, orig_mask, 3);

    // Create the inverted mask for the mustache
    Mat orig_mask_inv;
    bitwise_not(orig_mask, orig_mask_inv);

    // Convert mustache image to BGR
    // and save the original image size (used later when resizing the image)
    imgMustache = imgMustache(Rect(0, 0, imgMustache.cols, imgMustache.rows));
    cvtColor(imgMustache, imgMustache, COLOR_BGRA2BGR);
    int origMustacheHeight = imgMustache.rows;
    int origMustacheWidth = imgMustache.cols;
    
    for (const auto& face : faces)
    {
        //Point center(face.x + face.width / 2, face.y + face.height / 2);
        rectangle(frame, face.tl(), face.br(), Scalar(0, 0, 255), 3);      

        Mat roi_gray = frame_gray(Rect(face.x, face.y, face.width, face.height));
        Mat roi_color = frame(Rect(face.x, face.y, face.width, face.height));

        // Detect a nose within the region bounded by each face (the ROI)
        vector<Rect> nose;        
        nose_cascade.detectMultiScale(roi_gray, nose);

        for (const auto& n : nose) {
            // Un-comment the next line for debug (draw box around the nose)
            // rectangle(roi_color, n, Scalar(255, 0, 0), 2);

            // The mustache should be 2 times the width of the nose
            int mustacheWidth = 2 * n.width;
            int mustacheHeight = mustacheWidth * origMustacheHeight / origMustacheWidth;

            // Center the mustache on the bottom of the nose
            int x1 = n.x - (mustacheWidth / 4);
            int x2 = n.x + n.width + (mustacheWidth / 4);
            int y1 = n.y + n.height - (mustacheHeight / 2);
            int y2 = n.y + n.height + (mustacheHeight / 2);

            // Check limits
            if (x1 < 0) x1 = 0;
            if (y1 < 0) y1 = 0;
            if (x2 > face.width)
                x2 = face.width;
            if (y2 > face.height) 
                y2 = face.height;

            // Re-calculate the width and height of the mustache image
            mustacheWidth = x2 - x1;
            mustacheHeight = y2 - y1;

            // Resize the original image and the masks to the mustache sizes
            // calculated above
            Mat mustache, mask, mask_inv;
            resize(imgMustache, mustache, Size(mustacheWidth, mustacheHeight), 0, 0, INTER_AREA);
            resize(orig_mask, mask, Size(mustacheWidth, mustacheHeight), 0, 0, INTER_AREA);
            resize(orig_mask_inv, mask_inv, Size(mustacheWidth, mustacheHeight), 0, 0, INTER_AREA);

            // take ROI for mustache from background equal to size of mustache image
            Mat roi = roi_color(Rect(x1, y1, mustacheWidth, mustacheHeight));

            // roi_bg contains the original image only where the mustache is not
            // in the region that is the size of the mustache.
            Mat roi_bg;
            bitwise_and(roi, roi, roi_bg, mask_inv);

            // roi_fg contains the image of the mustache only where the mustache is
            Mat roi_fg;
            bitwise_and(mustache, mustache, roi_fg, mask);

            // join the roi_bg and roi_fg
            Mat overlay;
            add(roi_bg, roi_fg, overlay);

            // place overlay image over the original image            
            overlay.copyTo(roi);

            break;
        }
    }

    //-- Show merged videoframe
    imshow("Capture - Face detection", frame);
}
