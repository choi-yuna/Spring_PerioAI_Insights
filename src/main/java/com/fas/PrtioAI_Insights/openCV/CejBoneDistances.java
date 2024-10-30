package com.fas.PrtioAI_Insights.openCV;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class CejBoneDistances {
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
            parseIniFile("C:/Users/fasol/OneDrive/바탕 화면/BRM 701~800/Labelling/draw/A_7_0701_01.ini");
            drawTeethMasks();
            drawAndMapCejMask();
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

            // 최소 Y 좌표와 CEJ 및 Bone 좌표 간의 거리와 바뀐 좌표 출력
            calculateAndPrintAdjustedCejBoneDistances();

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
        cejMappedOnlyMask = Mat.zeros(3000, 3000, CvType.CV_8UC3);
        boneMappedOnlyMask = Mat.zeros(3000, 3000, CvType.CV_8UC3);

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
                type_ = "D";
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

    private static void calculateAndPrintAdjustedCejBoneDistances() {
        // 각 치아의 최소 Y 좌표 계산
        Map<Integer, Double> minYByTooth = new HashMap<>();

        for (int i = 0; i < teethPoints.size(); i++) {
            int toothNum = teethNum.get(i);
            List<Point> points = teethPoints.get(i);

            for (Point p : points) {
                if (!minYByTooth.containsKey(toothNum) || p.y < minYByTooth.get(toothNum)) {
                    minYByTooth.put(toothNum, p.y);
                }
            }
        }

        // CEJ와 Bone 좌표 각각에 대해 최소 Y 좌표에 맞춘 새로운 좌표와의 거리 계산
        for (Map.Entry<Integer, List<Point>> entry : teethCejPoints.entrySet()) {
            int toothNum = entry.getKey();
            List<Point> cejList = entry.getValue();

            if (minYByTooth.containsKey(toothNum)) {
                double minY = minYByTooth.get(toothNum);

                System.out.println("치아 번호: " + toothNum + " (CEJ와 최소 Y 기준)");
                for (Point cejPoint : cejList) {
                    double adjustedY = cejPoint.y - minY;
                    Point adjustedCej = new Point(cejPoint.x, adjustedY);
                    double distance = Math.abs(adjustedY);
                    System.out.println("    CEJ 원본 좌표: " + cejPoint + " -> 조정된 좌표: " + adjustedCej + " - 최소 Y와의 거리: " + distance);
                }
            }
        }

        for (Map.Entry<Integer, List<Point>> entry : bonePointsByNum.entrySet()) {
            int toothNum = entry.getKey();
            List<Point> boneList = entry.getValue();

            if (minYByTooth.containsKey(toothNum)) {
                double minY = minYByTooth.get(toothNum);

                System.out.println("치아 번호: " + toothNum + " (Bone과 최소 Y 기준)");
                for (Point bonePoint : boneList) {
                    double adjustedY = bonePoint.y - minY;
                    Point adjustedBone = new Point(bonePoint.x, adjustedY);
                    double distance = Math.abs(adjustedY);
                    System.out.println("    Bone 원본 좌표: " + bonePoint + " -> 조정된 좌표: " + adjustedBone + " - 최소 Y와의 거리: " + distance);
                }
            }
        }
    }
    private static void drawTeethMasks() {
        Map<Integer, Double> minYByTooth = new HashMap<>();
        Map<Integer, Double> maxYByTooth = new HashMap<>();
        Map<Integer, Double> minXByTooth = new HashMap<>();
        Map<Integer, Double> maxXByTooth = new HashMap<>();

        Map<Integer, Point> minPointYByTooth = new HashMap<>();
        Map<Integer, Point> maxPointYByTooth = new HashMap<>();
        Map<Integer, Point> minPointXByTooth = new HashMap<>();
        Map<Integer, Point> maxPointXByTooth = new HashMap<>();

        for (int i = 0; i < teethPoints.size(); i++) {
            int toothNum = teethNum.get(i);
            if (toothNum < 11 || toothNum > 48) continue;

            List<Point> points = teethPoints.get(i);
            if (points.size() < 3) continue;

            MatOfPoint pts = new MatOfPoint();
            pts.fromList(points);
            int thickness = teethSize.get(i);

            double area = Imgproc.contourArea(pts);
            if (area < 500) continue;

            // Teeth 폴리곤 그리기
            Imgproc.polylines(combinedMask, List.of(pts), true, new Scalar(255, 255, 255), thickness);
            Imgproc.fillPoly(combinedMask, List.of(pts), new Scalar(255, 255, 255));

            // 각 폴리곤 내의 최소 및 최대 x, y 좌표 찾기
            for (Point p : points) {
                // 최소 y값 갱신
                if (!minYByTooth.containsKey(toothNum) || p.y < minYByTooth.get(toothNum)) {
                    minYByTooth.put(toothNum, p.y);
                    minPointYByTooth.put(toothNum, p);
                }
                // 최대 y값 갱신
                if (!maxYByTooth.containsKey(toothNum) || p.y > maxYByTooth.get(toothNum)) {
                    maxYByTooth.put(toothNum, p.y);
                    maxPointYByTooth.put(toothNum, p);
                }
                // 최소 x값 갱신
                if (!minXByTooth.containsKey(toothNum) || p.x < minXByTooth.get(toothNum)) {
                    minXByTooth.put(toothNum, p.x);
                    minPointXByTooth.put(toothNum, p);
                }
                // 최대 x값 갱신
                if (!maxXByTooth.containsKey(toothNum) || p.x > maxXByTooth.get(toothNum)) {
                    maxXByTooth.put(toothNum, p.x);
                    maxPointXByTooth.put(toothNum, p);
                }
            }
        }

        // 치아 번호별 최대 및 최소 x, y 좌표 출력
        for (int toothNum : minYByTooth.keySet()) {
            System.out.println("치아 번호: " + toothNum +
                    " - 최소 y 좌표: " + minPointYByTooth.get(toothNum) +
                    ", 최대 y 좌표: " + maxPointYByTooth.get(toothNum) +
                    ", 최소 x 좌표: " + minPointXByTooth.get(toothNum) +
                    ", 최대 x 좌표: " + maxPointXByTooth.get(toothNum));
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

                // 특정 Y 좌표 범위에 있는 포인트만 허용 (예: Y 범위 필터링)
                int minY = toothBoundingBox.y - 50; // 여유값 추가
                int maxY = toothBoundingBox.y + toothBoundingBox.height + 50;

                for (Point cejPoint : points) {
                    if (toothBoundingBox.contains(cejPoint) &&
                            cejPoint.y >= minY && cejPoint.y <= maxY) { // Y 좌표 필터링 조건 추가
                        int toothNum = teethNum.get(j);
                        teethCejPoints.computeIfAbsent(toothNum, k -> new ArrayList<>()).add(cejPoint);
                        Imgproc.circle(cejMappedOnlyMask, cejPoint, 2, new Scalar(0, 255, 0), -1);
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

                // Y 좌표 범위 필터링 설정 (치아와 연관된 영역만 허용)
                int minY = toothBoundingBox.y - 50; // 여유값 추가
                int maxY = toothBoundingBox.y + toothBoundingBox.height + 50;


                for (Point cejPoint : points) {
                    if (toothBoundingBox.contains(cejPoint) &&
                            cejPoint.y >= minY && cejPoint.y <= maxY) { // Y 좌표 필터링 조건 추가
                        int toothNum = teethNum.get(j);
                        bonePointsByNum.computeIfAbsent(toothNum, k -> new ArrayList<>()).add(cejPoint);
                        Imgproc.circle(boneMappedOnlyMask, cejPoint, 3, new Scalar(0, 255, 0), -1);
                    }
                }
            }
        }
    }

    private static void removeIslands(Map<String, Mat> bimasks, int minArea) {
        for (Map.Entry<String, Mat> entry : bimasks.entrySet()) {
            Mat bimask = entry.getValue();

            // connectedComponentsWithStats를 사용하여 라벨링 및 컴포넌트 정보 추출
            Mat labels = new Mat();
            Mat stats = new Mat();
            Mat centroids = new Mat();
            int numLabels = Imgproc.connectedComponentsWithStats(bimask, labels, stats, centroids);

            // 각 라벨에 대해 최소 면적(minArea) 이하인 작은 컴포넌트는 제거
            for (int i = 1; i < numLabels; i++) { // 라벨 0은 배경이므로 제외
                int area = (int) stats.get(i, Imgproc.CC_STAT_AREA)[0];
                if (area <= minArea) {
                    // 면적이 minArea 이하인 컴포넌트를 제거
                    Core.compare(labels, new Scalar(i), bimask, Core.CMP_NE);
                }
            }

            // 수정된 bimask를 맵에 다시 저장
            entry.setValue(bimask);
        }
    }
}
