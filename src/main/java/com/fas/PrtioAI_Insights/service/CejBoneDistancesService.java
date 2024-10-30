package com.fas.PrtioAI_Insights.service;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgcodecs.Imgcodecs;
import org.springframework.stereotype.Service;

import java.io.*;
import java.util.*;

@Service
public class CejBoneDistancesService {

    // 데이터 초기화
    private List<Integer> teethNum = new ArrayList<>();
    private List<List<Point>> teethPoints = new ArrayList<>();
    private List<Integer> teethSize = new ArrayList<>();
    private List<List<Point>> cejPoints = new ArrayList<>();
    private List<Scalar> cejColor = new ArrayList<>();
    private List<Integer> cejSize = new ArrayList<>();

    // 추가 데이터 초기화
    private List<List<Point>> tlaPoints = new ArrayList<>();
    private List<Scalar> tlaColor = new ArrayList<>();
    private List<Integer> tlaSize = new ArrayList<>();
    private List<List<Point>> bonePoints = new ArrayList<>();
    private List<Scalar> boneColor = new ArrayList<>();
    private List<Integer> boneSize = new ArrayList<>();

    // 치아 번호별 매핑
    private Map<Integer, List<Point>> teethCejPoints = new HashMap<>();
    private Map<Integer, List<List<Point>>> tlaPointsByNum = new HashMap<>();
    private Map<Integer, List<Point>> bonePointsByNum = new HashMap<>();
    private Map<String, Mat> bimasks = new HashMap<>();  // 각 치아 번호에 대한 마스크

    // 마스크
    private Mat combinedMask;
    private Mat cejMask;
    private Mat mappedCejMask;
    private Mat tlaMask;
    private Mat boneMask;
    private Mat cejMappedOnlyMask;
    private Mat boneMappedOnlyMask;

    public CejBoneDistancesService() {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        initializeMasks();
    }

    private void initializeMasks() {
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

    public void parseIniFile(String filepath) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(filepath));
        System.out.println("Parsing INI file: " + filepath);
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

        // CEJ와 Bone 데이터 매핑
        drawAndMapCejMask();
        drawAndMapBoneMask();

        // 분석 데이터 반환
        Map<String, Object> analysisData = getAnalysisData();
    }

    public Map<Integer, Map<String, Object>> calculateAdjustedCejBoneDistances() {
        Map<Integer, Map<String, Object>> result = new HashMap<>();
        Map<Integer, Double> minYByTooth = new HashMap<>();

        // 각 치아의 최소 Y 좌표 계산
        for (int i = 0; i < teethPoints.size(); i++) {
            int toothNum = teethNum.get(i);
            List<Point> points = teethPoints.get(i);

            for (Point p : points) {
                minYByTooth.put(toothNum, Math.min(minYByTooth.getOrDefault(toothNum, Double.MAX_VALUE), p.y));
            }
        }

        // CEJ 데이터 저장
        for (Map.Entry<Integer, List<Point>> entry : teethCejPoints.entrySet()) {
            int toothNum = entry.getKey();
            List<Point> cejList = entry.getValue();

            Map<String, Object> toothData = result.computeIfAbsent(toothNum, k -> new HashMap<>());
            List<Map<String, Double>> adjustedCejPoints = new ArrayList<>();
            List<Double> cejDistances = new ArrayList<>();

            if (minYByTooth.containsKey(toothNum)) {
                double minY = minYByTooth.get(toothNum);

                for (Point cejPoint : cejList) {
                    double adjustedY = cejPoint.y - minY;
                    adjustedCejPoints.add(Map.of("x", cejPoint.x, "y", adjustedY));
                    cejDistances.add(Math.abs(adjustedY));
                }
            }

            toothData.put("cejPoints", cejList);  // 원본 CEJ 좌표
            toothData.put("adjustedCejPoints", adjustedCejPoints);  // 조정된 CEJ 좌표
            toothData.put("cejDistances", cejDistances);  // CEJ 거리
        }

        // Bone 데이터 저장
        for (Map.Entry<Integer, List<Point>> entry : bonePointsByNum.entrySet()) {
            int toothNum = entry.getKey();
            List<Point> boneList = entry.getValue();

            Map<String, Object> toothData = result.computeIfAbsent(toothNum, k -> new HashMap<>());
            List<Map<String, Double>> adjustedBonePoints = new ArrayList<>();
            List<Double> boneDistances = new ArrayList<>();

            if (minYByTooth.containsKey(toothNum)) {
                double minY = minYByTooth.get(toothNum);

                for (Point bonePoint : boneList) {
                    double adjustedY = bonePoint.y - minY;
                    adjustedBonePoints.add(Map.of("x", bonePoint.x, "y", adjustedY));
                    boneDistances.add(Math.abs(adjustedY));
                }
            }

            toothData.put("bonePoints", boneList);  // 원본 Bone 좌표
            toothData.put("adjustedBonePoints", adjustedBonePoints);  // 조정된 Bone 좌표
            toothData.put("boneDistances", boneDistances);  // Bone 거리
        }

        return result;
    }

    public void drawTeethMasks() {
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

            Imgproc.polylines(combinedMask, List.of(pts), true, new Scalar(255, 255, 255), thickness);
            Imgproc.fillPoly(combinedMask, List.of(pts), new Scalar(255, 255, 255));

            for (Point p : points) {
                if (!minYByTooth.containsKey(toothNum) || p.y < minYByTooth.get(toothNum)) {
                    minYByTooth.put(toothNum, p.y);
                    minPointYByTooth.put(toothNum, p);
                }
                if (!maxYByTooth.containsKey(toothNum) || p.y > maxYByTooth.get(toothNum)) {
                    maxYByTooth.put(toothNum, p.y);
                    maxPointYByTooth.put(toothNum, p);
                }
                if (!minXByTooth.containsKey(toothNum) || p.x < minXByTooth.get(toothNum)) {
                    minXByTooth.put(toothNum, p.x);
                    minPointXByTooth.put(toothNum, p);
                }
                if (!maxXByTooth.containsKey(toothNum) || p.x > maxXByTooth.get(toothNum)) {
                    maxXByTooth.put(toothNum, p.x);
                    maxPointXByTooth.put(toothNum, p);
                }
            }
        }

        for (int toothNum : minYByTooth.keySet()) {
            System.out.println("치아 번호: " + toothNum +
                    " - 최소 y 좌표: " + minPointYByTooth.get(toothNum) +
                    ", 최대 y 좌표: " + maxPointYByTooth.get(toothNum) +
                    ", 최소 x 좌표: " + minPointXByTooth.get(toothNum) +
                    ", 최대 x 좌표: " + maxPointXByTooth.get(toothNum));
        }
    }

    private void drawAndMapCejMask() {
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

    private void drawAndMapBoneMask() {
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

    public void removeIslands(int minArea) {
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

    public void saveMasks() {
        Imgcodecs.imwrite("Combined_Teeth_Mask.png", combinedMask);
        Imgcodecs.imwrite("cejMask.png", cejMask);
        Imgcodecs.imwrite("mappedCejMask.png", mappedCejMask);
        Imgcodecs.imwrite("tlaMask.png", tlaMask);
        Imgcodecs.imwrite("boneMask.png", boneMask);
        Imgcodecs.imwrite("cejMappedOnly.png", cejMappedOnlyMask);
        Imgcodecs.imwrite("boneMappedOnly.png", boneMappedOnlyMask);
    }

    public Map<String, Object> getAnalysisData() {
        Map<String, Object> result = new HashMap<>();
        result.put("teethNum", teethNum);
        result.put("teethPoints", teethPoints);
        result.put("teethSize", teethSize);
        result.put("cejPoints", cejPoints);
        result.put("cejColor", cejColor);
        result.put("cejSize", cejSize);
        result.put("tlaPoints", tlaPoints);
        result.put("tlaColor", tlaColor);
        result.put("tlaSize", tlaSize);
        result.put("bonePoints", bonePoints);
        result.put("boneColor", boneColor);
        result.put("boneSize", boneSize);
        result.put("teethCejPoints", teethCejPoints);
        result.put("tlaPointsByNum", tlaPointsByNum);
        result.put("bonePointsByNum", bonePointsByNum);
        return result;
    }
}
