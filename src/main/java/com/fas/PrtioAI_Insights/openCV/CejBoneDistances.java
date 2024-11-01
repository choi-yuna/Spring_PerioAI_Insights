package com.fas.PrtioAI_Insights.openCV;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.*;
import java.util.*;

public class CejBoneDistances {
    // 데이터 초기화
    private static List<Integer> teethNum = new ArrayList<>();
    private static List<List<Point>> teethPoints = new ArrayList<>();
    private static List<Integer> teethSize = new ArrayList<>();
    private static List<List<Point>> cejPoints = new ArrayList<>();
    private static List<Scalar> cejColor = new ArrayList<>();
    private static List<Integer> cejSize = new ArrayList<>();

    private static List<List<Point>> tlaPoints = new ArrayList<>();
    private static List<Scalar> tlaColor = new ArrayList<>();
    private static List<Integer> tlaSize = new ArrayList<>();
    private static List<List<Point>> bonePoints = new ArrayList<>();
    private static List<Scalar> boneColor = new ArrayList<>();
    private static List<Integer> boneSize = new ArrayList<>();

    private static Map<Integer, List<Point>> teethCejPoints = new HashMap<>();
    private static Map<Integer, List<List<Point>>> tlaPointsByNum = new HashMap<>();
    private static Map<Integer, List<Point>> bonePointsByNum = new HashMap<>();
    private static Map<Integer, Double> yReferenceByTooth = new HashMap<>();
    private static Map<String, Mat> bimasks = new HashMap<>();

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
            drawCombinedCejAndBoneMask(); // 필터링된 CEJ와 Bone을 한번에 그리는 함수 호출

            removeIslands(bimasks, 900);

            Imgcodecs.imwrite("Combined_Teeth_Mask.png", combinedMask);
            Imgcodecs.imwrite("cejMask.png", cejMask);
            Imgcodecs.imwrite("mappedCejMask.png", mappedCejMask);
            Imgcodecs.imwrite("tlaMask.png", tlaMask);
            Imgcodecs.imwrite("boneMask.png", boneMask);
            Imgcodecs.imwrite("cejMappedOnly.png", cejMappedOnlyMask);
            Imgcodecs.imwrite("boneMappedOnly.png", boneMappedOnlyMask);

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

    // 필터링된 CEJ 및 Bone 좌표를 한 번에 그려주는 함수
    private static void drawCombinedCejAndBoneMask() {
        for (Map.Entry<Integer, List<Point>> entry : teethCejPoints.entrySet()) {
            int toothNum = entry.getKey();
            List<Point> cejFilteredPoints = entry.getValue();

            if (cejFilteredPoints.size() >= 3) {
                MatOfPoint pts = new MatOfPoint();
                pts.fromList(cejFilteredPoints);
                Imgproc.polylines(cejMask, List.of(pts), true, new Scalar(0, 0, 255), 2);
                Imgproc.fillPoly(cejMask, List.of(pts), new Scalar(0, 0, 255));
            }
        }

        for (Map.Entry<Integer, List<Point>> entry : bonePointsByNum.entrySet()) {
            int toothNum = entry.getKey();
            List<Point> boneFilteredPoints = entry.getValue();

            if (boneFilteredPoints.size() >= 3) {
                MatOfPoint pts = new MatOfPoint();
                pts.fromList(boneFilteredPoints);
                Imgproc.polylines(boneMask, List.of(pts), true, new Scalar(0, 255, 0), 2);
                Imgproc.fillPoly(boneMask, List.of(pts), new Scalar(0, 255, 0));
            }
        }
    }

    private static void drawTeethMasks() {
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

            Imgproc.polylines(combinedMask, List.of(pts), true, new Scalar(255, 255, 255), thickness);
            Imgproc.fillPoly(combinedMask, List.of(pts), new Scalar(255, 255, 255));

            double minY = Double.MAX_VALUE;
            double maxY = Double.MIN_VALUE;

            for (Point p : points) {
                if (p.y < minY) minY = p.y;
                if (p.y > maxY) maxY = p.y;
            }

            // 상악은 maxY, 하악은 minY를 기준 Y 좌표로 저장
            if (toothNum >= 11 && toothNum <= 28) {
                yReferenceByTooth.put(toothNum, maxY); // 상악 최대 Y
            } else if (toothNum >= 31 && toothNum <= 48) {
                yReferenceByTooth.put(toothNum, minY); // 하악 최소 Y
            }
        }
    }

    private static void calculateAndPrintAdjustedCejBoneDistances() {
        // 상악 치아와 하악 치아 번호 구분
        Set<Integer> maxillaryTeeth = Set.of(11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28);
        Set<Integer> mandibularTeeth = Set.of(31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48);

        System.out.println("Y 기준 좌표 (yReferenceByTooth): " + yReferenceByTooth);

        // CEJ 좌표 계산
        for (Map.Entry<Integer, List<Point>> entry : teethCejPoints.entrySet()) {
            int toothNum = entry.getKey();
            List<Point> cejList = entry.getValue();

            if (yReferenceByTooth.containsKey(toothNum)) {
                double yReference = yReferenceByTooth.get(toothNum);
                System.out.println("치아 번호: " + toothNum + " - Y 기준 좌표: " + yReference);

                for (Point cejPoint : cejList) {
                    double adjustedY;

                    if (maxillaryTeeth.contains(toothNum)) {
                        // 상악 치아: 기준 Y에서 현재 Y를 뺌
                        adjustedY = yReference - cejPoint.y;
                    } else {
                        // 하악 치아: 현재 Y에서 기준 Y를 뺌
                        adjustedY = cejPoint.y - yReference;
                    }

                    Point adjustedCej = new Point(cejPoint.x, adjustedY);
                    double distance = Math.abs(adjustedY);
                    System.out.println("    CEJ 원본 좌표: " + cejPoint + " -> 조정된 좌표: " + adjustedCej + " - 기준 Y와의 거리: " + distance);
                }
            }
        }

        // Bone 좌표 계산
        for (Map.Entry<Integer, List<Point>> entry : bonePointsByNum.entrySet()) {
            int toothNum = entry.getKey();
            List<Point> boneList = entry.getValue();

            if (yReferenceByTooth.containsKey(toothNum)) {
                double yReference = yReferenceByTooth.get(toothNum);
                System.out.println("치아 번호: " + toothNum + " - Y 기준 좌표: " + yReference);

                for (Point bonePoint : boneList) {
                    double adjustedY;

                    if (maxillaryTeeth.contains(toothNum)) {
                        // 상악 치아: 기준 Y에서 현재 Y를 뺌
                        adjustedY = yReference - bonePoint.y;
                    } else {
                        // 하악 치아: 현재 Y에서 기준 Y를 뺌
                        adjustedY = bonePoint.y - yReference;
                    }

                    Point adjustedBone = new Point(bonePoint.x, adjustedY);
                    double distance = Math.abs(adjustedY);
                    System.out.println("    Bone 원본 좌표: " + bonePoint + " -> 조정된 좌표: " + adjustedBone + " - 기준 Y와의 거리: " + distance);
                }
            }
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
