package com.fas.PrtioAI_Insights.openCV;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class OpenCVTeethMask {
    // 데이터 초기화
    private static List<Integer> teethNum = new ArrayList<>();
    private static List<List<Point>> teethPoints = new ArrayList<>();
    private static List<Integer> teethSize = new ArrayList<>();
    private static List<List<Point>> cejPoints = new ArrayList<>();
    private static List<Scalar> cejColor = new ArrayList<>();
    private static List<Integer> cejSize = new ArrayList<>();

    // 추가 데이터 초기화
    private static List<List<Point>> tlaPoints = new ArrayList<>();
    private static List<Scalar> tlaColor = new ArrayList<>();
    private static List<Integer> tlaSize = new ArrayList<>();
    private static List<List<Point>> bonePoints = new ArrayList<>();
    private static List<Scalar> boneColor = new ArrayList<>();
    private static List<Integer> boneSize = new ArrayList<>();

    // 치아 번호별 매핑
    private static Map<Integer, List<Point>> teethCejPoints = new HashMap<>();
    private static Map<Integer, List<List<Point>>> tlaPointsByNum = new HashMap<>();
    private static Map<Integer, List<Point>> bonePointsByNum = new HashMap<>();
    private static Map<String, Mat> bimasks = new HashMap<>();  // 각 치아 번호에 대한 마스크

    // 마스크
    private static Mat combinedMask;
    private static Mat cejMask;
    private static Mat mappedCejMask;
    private static Mat tlaMask;
    private static Mat boneMask;
    private static Mat cejMappedOnlyMask;
    private static Mat boneMappedOnlyMask;

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // 마스크 초기화
        initializeMasks();

        try {
            parseIniFile("C:/Users/wlsdn/Desktop/BRM 701~800/Labelling/draw/A_7_0704_01.ini");
            drawTeethMasks();
            drawAndMapCejMask();
            drawTlaMask();
            drawAndMapBoneMask();

            // 작은 섬 제거 (면적이 900 이하인 컴포넌트)
            removeIslands(bimasks, 900);

            // 결과 이미지 저장
            Imgcodecs.imwrite("Combined_Teeth_Mask.png", combinedMask);
            Imgcodecs.imwrite("cejMask.png", cejMask);
            Imgcodecs.imwrite("mappedCejMask.png", mappedCejMask);
            Imgcodecs.imwrite("tlaMask.png", tlaMask);
            Imgcodecs.imwrite("boneMask.png", boneMask);
            Imgcodecs.imwrite("cejMappedOnly.png", cejMappedOnlyMask);  // 매핑된 CEJ 좌표만 저장
            Imgcodecs.imwrite("boneMappedOnly.png", boneMappedOnlyMask);  // 매핑된 Bone 좌표만 저장

            // 치아 번호별 CEJ 및 Bone 좌표 출력
            for (Map.Entry<Integer, List<Point>> entry : teethCejPoints.entrySet()) {
                System.out.println("치아 번호: " + entry.getKey() + " - CEJ 좌표: " + entry.getValue());
            }
            for (Map.Entry<Integer, List<Point>> entry : bonePointsByNum.entrySet()) {
                System.out.println("치아 번호: " + entry.getKey() + " - Bone 좌표: " + entry.getValue());
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void initializeMasks() {
        combinedMask = Mat.zeros(3000, 3000, CvType.CV_8UC3);
        cejMask = Mat.zeros(3000, 3000, CvType.CV_8UC3);
        mappedCejMask = Mat.zeros(3000, 3000, CvType.CV_8UC3);
        tlaMask = Mat.zeros(3000, 3000, CvType.CV_8UC3);
        boneMask = Mat.zeros(3000, 3000, CvType.CV_8UC3);
        cejMappedOnlyMask = Mat.zeros(3000, 3000, CvType.CV_8UC3);  // 매핑된 CEJ만 저장할 마스크
        boneMappedOnlyMask = Mat.zeros(3000, 3000, CvType.CV_8UC3);  // 매핑된 Bone만 저장할 마스크

        // 각 치아 번호에 빈 마스크 초기화
        for (int i = 11; i <= 48; i++) {
            bimasks.put(String.valueOf(i), Mat.zeros(3000, 3000, CvType.CV_8UC1));
        }
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
                    } else if ("A".equals(type_)) {
                        tlaPoints.add(new ArrayList<>(loadedPoints));
                        tlaColor.add(new Scalar(loadedColor.get(0), loadedColor.get(1), loadedColor.get(2), loadedColor.get(3)));
                        tlaSize.add(_Size);
                        tlaPointsByNum.computeIfAbsent(num, k -> new ArrayList<>()).add(new ArrayList<>(loadedPoints));
                    } else if ("D".equals(type_)) {
                        bonePoints.add(new ArrayList<>(loadedPoints));
                        boneColor.add(new Scalar(loadedColor.get(0), loadedColor.get(1), loadedColor.get(2), loadedColor.get(3)));
                        boneSize.add(_Size);
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
            } else if (work.equals("S") && line.startsWith("BD")) {
                type_ = "D"; // 치주골 데이터를 위한 타입
            } else if (work.equals("S") && line.startsWith("AD")) {
                type_ = "A";
            } else if (work.equals("S") && line.startsWith("DD")) {
                type_ = "D";
            } else if (line.startsWith("C=")) {
                String[] parts = line.substring(2).split(",");
                for (String part : parts) {
                    loadedColor.add(Integer.parseInt(part.trim()));
                }
            } else if (line.startsWith("P=")) {
                String[] parts = line.substring(2).split(",");
                int x = Integer.parseInt(parts[0].trim());
                int y = Integer.parseInt(parts[1].trim());
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

    private static void drawTeethMasks() {
        for (int i = 0; i < teethPoints.size(); i++) {
            if (teethNum.get(i) < 11 || teethNum.get(i) > 48) continue;

            List<Point> points = teethPoints.get(i);
            if (points.size() < 3) continue;

            MatOfPoint pts = new MatOfPoint();
            pts.fromList(points);
            int thickness = teethSize.get(i);

            double area = Imgproc.contourArea(pts);
            if (area < 500) continue;

            Imgproc.polylines(combinedMask, List.of(pts), true, new Scalar(255, 255, 255), thickness);
            Imgproc.fillPoly(combinedMask, List.of(pts), new Scalar(255, 255, 255));
        }
    }

    private static void drawAndMapCejMask() {
        for (int i = 0; i < cejPoints.size(); i++) {
            List<Point> points = cejPoints.get(i);
            if (points.size() < 3) continue;

            MatOfPoint pts = new MatOfPoint();
            pts.fromList(points);
            int thickness = cejSize.get(i);

            double area = Imgproc.contourArea(pts);
            if (area < 300 || thickness > 2) continue;

            for (int j = 0; j < teethPoints.size(); j++) {
                if (teethNum.get(j) < 11 || teethNum.get(j) > 48) continue;

                List<Point> toothPoints = teethPoints.get(j);
                MatOfPoint toothPts = new MatOfPoint();
                toothPts.fromList(toothPoints);

                Rect toothBoundingBox = Imgproc.boundingRect(toothPts);

                int minY = toothBoundingBox.y - 50;
                int maxY = toothBoundingBox.y + toothBoundingBox.height + 50;

                for (Point cejPoint : points) {
                    if (toothBoundingBox.contains(cejPoint) &&
                            cejPoint.y >= minY && cejPoint.y <= maxY) {
                        int toothNum = teethNum.get(j);
                        Point normalizedPoint = normalizePoint(cejPoint, toothNum); // 호출 수정
                        teethCejPoints.computeIfAbsent(toothNum, k -> new ArrayList<>()).add(normalizedPoint);
                        Imgproc.circle(cejMappedOnlyMask, normalizedPoint, 2, new Scalar(0, 255, 0), -1);
                    }
                }
            }
        }
    }

    private static void drawAndMapBoneMask() {
        for (int i = 0; i < bonePoints.size(); i++) {
            List<Point> points = bonePoints.get(i);
            if (points.size() < 3) continue;

            MatOfPoint pts = new MatOfPoint();
            pts.fromList(points);
            int thickness = boneSize.get(i);

            double area = Imgproc.contourArea(pts);
            if (area < 300 || thickness > 2) continue;

            for (int j = 0; j < teethPoints.size(); j++) {
                if (teethNum.get(j) < 11 || teethNum.get(j) > 48) continue;

                List<Point> toothPoints = teethPoints.get(j);
                MatOfPoint toothPts = new MatOfPoint();
                toothPts.fromList(toothPoints);

                Rect toothBoundingBox = Imgproc.boundingRect(toothPts);

                int minY = toothBoundingBox.y - 50;
                int maxY = toothBoundingBox.y + toothBoundingBox.height + 50;

                for (Point bonePoint : points) {
                    if (toothBoundingBox.contains(bonePoint) &&
                            bonePoint.y >= minY && bonePoint.y <= maxY) {
                        int toothNum = teethNum.get(j);
                        Point normalizedPoint = normalizePoint(bonePoint, toothNum); // 호출 수정
                        bonePointsByNum.computeIfAbsent(toothNum, k -> new ArrayList<>()).add(normalizedPoint);
                        Imgproc.circle(boneMappedOnlyMask, normalizedPoint, 3, new Scalar(0, 255, 0), -1);
                    }
                }
            }
        }
    }

    private static void drawTlaMask() {
        for (Map.Entry<Integer, List<List<Point>>> entry : tlaPointsByNum.entrySet()) {
            for (List<Point> points : entry.getValue()) {
                MatOfPoint pts = new MatOfPoint();
                pts.fromList(points);
                Imgproc.polylines(tlaMask, List.of(pts), true, new Scalar(0, 0, 255), 2);
            }
        }
    }

    private static Point normalizePoint(Point originalPoint, int toothNum) {
        // y축 정규화: 원래 비율을 유지하며 0 ~ 2로 변환
        double normalizedY = (originalPoint.y / 3000.0) * 2.0;

        // x축 정규화: 상악과 하악의 치아 순서에 맞춰 0 ~ 48 범위로 비율 유지하며 변환
        double normalizedX;
        if (toothNum >= 11 && toothNum <= 28) { // 상악 치아 (11번 ~ 28번)
            int[] maxillaryOrder = {18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28};
            int index = findToothIndex(toothNum, maxillaryOrder);
            double toothPosition = index / 15.0; // 0 ~ 1 사이 위치 비율
            normalizedX = toothPosition * 48.0 + (originalPoint.x / 3000.0) * (48.0 / 15.0);
        } else if (toothNum >= 31 && toothNum <= 48) { // 하악 치아 (31번 ~ 48번)
            int[] mandibularOrder = {48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38};
            int index = findToothIndex(toothNum, mandibularOrder);
            double toothPosition = index / 15.0; // 0 ~ 1 사이 위치 비율
            normalizedX = toothPosition * 48.0 + (originalPoint.x / 3000.0) * (48.0 / 15.0);
        } else {
            normalizedX = originalPoint.x; // 치아 번호가 범위 밖일 경우 원래 값 유지
        }

        return new Point(normalizedX, normalizedY);
    }

    // 주어진 치아 번호 배열에서 현재 치아 번호의 인덱스를 찾는 메서드
    private static int findToothIndex(int toothNum, int[] toothOrder) {
        for (int i = 0; i < toothOrder.length; i++) {
            if (toothOrder[i] == toothNum) {
                return i;
            }
        }
        return -1; // 번호가 배열에 없을 경우 -1 반환
    }


    private static void removeIslands(Map<String, Mat> bimasks, int minArea) {
        for (Map.Entry<String, Mat> entry : bimasks.entrySet()) {
            Mat bimask = entry.getValue();

            Mat labels = new Mat();
            Mat stats = new Mat();
            Mat centroids = new Mat();
            int numLabels = Imgproc.connectedComponentsWithStats(bimask, labels, stats, centroids);

            for (int i = 1; i < numLabels; i++) {
                int area = (int) stats.get(i, Imgproc.CC_STAT_AREA)[0];
                if (area <= minArea) {
                    Core.compare(labels, new Scalar(i), bimask, Core.CMP_NE);
                }
            }
            entry.setValue(bimask);
        }
    }
}
