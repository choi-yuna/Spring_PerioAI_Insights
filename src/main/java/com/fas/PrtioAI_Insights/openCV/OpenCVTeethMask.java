package com.fas.PrtioAI_Insights.openCV;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class OpenCVTeethMask {
    // 데이터 초기화
    private static List<Integer> teethNum = new ArrayList<>();
    private static List<List<Point>> teethPoints = new ArrayList<>();
    private static List<Integer> teethSize = new ArrayList<>();
    private static List<List<Point>> cejPoints = new ArrayList<>();
    private static List<Scalar> cejColor = new ArrayList<>();
    private static List<Integer> cejSize = new ArrayList<>();

    // CEJ 마스크 및 전체 치아 마스크
    private static Mat combinedMask; // 치아 마스크를 그릴 큰 Mat 객체
    private static Mat cejMask; // CEJ 마스크를 별도로 그릴 Mat 객체

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // OpenCV 라이브러리 로드 후 마스크 초기화
        initializeMasks();

        try {
            parseIniFile("C:/Users/fasol/OneDrive/바탕 화면/BRM 701~800/Labelling/draw/A_7_0701_01.ini");
            drawCejMask();
            drawTeethMasks();

            // 결과 이미지 저장 (치아와 CEJ 마스크를 별도 이미지로 저장)
            Imgcodecs.imwrite("Combined_Teeth_Mask.png", combinedMask);
            Imgcodecs.imwrite("cejMask.png", cejMask);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void initializeMasks() {
        // combinedMask: 3000x3000 크기의 전체 치아 마스크 (흰색)
        combinedMask = Mat.zeros(3000, 3000, CvType.CV_8UC3);

        // cejMask: 3000x3000 크기의 CEJ 마스크 (빨간색)
        cejMask = Mat.zeros(3000, 3000, CvType.CV_8UC3);
    }

    private static void parseIniFile(String filepath) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(filepath));
        String line;
        List<Point> loadedPoints = new ArrayList<>();
        List<Integer> loadedColor = new ArrayList<>();
        int _Size = 0;
        boolean Rect = false;
        String type_ = "";
        String work = "";
        int num = 0;

        while ((line = br.readLine()) != null) {
            line = line.trim();
            if (line.startsWith("START")) {
                work = "S";
            } else if (line.startsWith("N=")) {
                num = Integer.parseInt(line.substring(2));
            } else if (line.startsWith("END")) {
                if (!Rect) {
                    if ("T".equals(type_)) {
                        teethPoints.add(new ArrayList<>(loadedPoints));
                        teethSize.add(_Size);
                        teethNum.add(num);
                    } else if ("C".equals(type_)) {
                        cejPoints.add(new ArrayList<>(loadedPoints));
                        cejColor.add(new Scalar(loadedColor.get(0), loadedColor.get(1), loadedColor.get(2), loadedColor.get(3)));
                        cejSize.add(_Size);
                    }
                }
                _Size = 0;
                loadedPoints.clear();
                loadedColor.clear();
                Rect = false;
            } else if (work.equals("S") && line.startsWith("TD")) {
                type_ = "T";
            } else if (work.equals("S") && line.startsWith("CD")) {
                type_ = "C";
            } else if (line.startsWith("C=")) {
                String[] parts = line.substring(2).split(",");
                for (String part : parts) {
                    loadedColor.add(Integer.parseInt(part.trim()));
                }
            } else if (line.startsWith("P=")) {
                String[] parts = line.substring(2).split(",");
                int x = Integer.parseInt(parts[0].trim());
                int y = Integer.parseInt(parts[1].trim());

                // 유효한 범위 내에 있는 좌표만 추가
                if (x >= 0 && x < 3000 && y >= 0 && y < 3000) {
                    loadedPoints.add(new Point(x, y));
                }
            } else if (line.startsWith("S=")) {
                _Size = Integer.parseInt(line.substring(2).trim());
            } else if (line.startsWith("R")) {
                Rect = true;
            }
        }
        br.close();
    }

    private static void drawCejMask() {
        for (int i = 0; i < cejPoints.size(); i++) {
            List<Point> points = cejPoints.get(i);
            if (points.size() < 3) {
                // 3개의 점 미만으로 구성된 폴리라인은 무시
                continue;
            }

            MatOfPoint pts = new MatOfPoint();
            pts.fromList(points);
            int thickness = cejSize.get(i);

            // 폴리곤의 면적이 일정 크기 이상인 경우에만 그리기
            double area = Imgproc.contourArea(pts);
            if (area < 300) { // 예: 면적이 300 이하인 작은 요소는 무시
                continue;
            }

            // 빨간색(0, 0, 255)으로 CEJ 폴리라인과 폴리곤 그리기
            Imgproc.polylines(cejMask, List.of(pts), false, new Scalar(0, 0, 255), thickness);
        }
    }

    private static void drawTeethMasks() {
        for (int i = 0; i < teethPoints.size(); i++) {
            if (teethNum.get(i) < 11 || teethNum.get(i) > 48) {
                // 유효한 치아 번호가 아닌 경우 건너뜁니다
                continue;
            }

            List<Point> points = teethPoints.get(i);
            if (points.size() < 3) {
                // 3개의 점 미만으로 구성된 폴리라인은 무시
                continue;
            }

            MatOfPoint pts = new MatOfPoint();
            pts.fromList(points);
            int thickness = teethSize.get(i);

            // 폴리곤의 면적이 일정 크기 이상인 경우에만 그리기
            double area = Imgproc.contourArea(pts);
            if (area < 500) { // 예: 면적이 500 이하인 작은 요소는 무시
                continue;
            }

            // 치아 폴리라인과 폴리곤을 흰색(255, 255, 255)으로 그리기
            Imgproc.polylines(combinedMask, List.of(pts), true, new Scalar(255, 255, 255), thickness);
            Imgproc.fillPoly(combinedMask, List.of(pts), new Scalar(255, 255, 255));
        }
    }
}
