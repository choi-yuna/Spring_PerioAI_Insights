package com.fas.PrtioAI_Insights;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;

public class Test {

    public static void main(String[] args) {
        // OpenCV 라이브러리 로드
        try {
            System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
            System.out.println("OpenCV 라이브러리가 성공적으로 로드되었습니다.");
        } catch (UnsatisfiedLinkError e) {
            System.err.println("OpenCV 라이브러리를 로드할 수 없습니다.");
            e.printStackTrace();
            return;
        }

        // 간단한 Mat 객체 생성
        Mat mat = Mat.eye(3, 3, CvType.CV_8UC1);
        System.out.println("3x3 단위 행렬 생성:");
        System.out.println(mat.dump());

        // 100x100 크기의 이미지 생성 및 초기화
        Mat img = new Mat(100, 100, CvType.CV_8UC3);
        for (int row = 0; row < img.rows(); row++) {
            for (int col = 0; col < img.cols(); col++) {
                img.put(row, col, 0, 255, 0); // 녹색으로 픽셀 설정
            }
        }

        // 이미지 파일로 저장
        Imgcodecs.imwrite("test_output.png", img);
        System.out.println("녹색 이미지를 test_output.png로 저장했습니다.");
        System.out.println("OpenCV 테스트 완료.");
    }
}
