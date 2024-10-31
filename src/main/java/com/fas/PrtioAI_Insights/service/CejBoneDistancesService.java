package com.fas.PrtioAI_Insights.service;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgcodecs.Imgcodecs;
import org.springframework.stereotype.Service;

import java.io.*;
import java.util.*;

@Service
public class CejBoneDistancesService {

    private List<Integer> teethNum;
    private List<List<Point>> teethPoints;
    private List<Integer> teethSize;
    private List<List<Point>> cejPoints;
    private List<Scalar> cejColor;
    private List<Integer> cejSize;
    private List<List<Point>> tlaPoints;
    private List<Scalar> tlaColor;
    private List<Integer> tlaSize;
    private List<List<Point>> bonePoints;
    private List<Scalar> boneColor;
    private List<Integer> boneSize;

    private Map<Integer, List<Point>> teethCejPoints;
    private Map<Integer, List<List<Point>>> tlaPointsByNum;
    private Map<Integer, List<Point>> bonePointsByNum;
    private Map<String, Mat> bimasks;

    private Mat combinedMask, cejMask, mappedCejMask, tlaMask, boneMask, cejMappedOnlyMask, boneMappedOnlyMask;

    public CejBoneDistancesService() {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    private void initialize() {
        teethNum = new ArrayList<>();
        teethPoints = new ArrayList<>();
        teethSize = new ArrayList<>();
        cejPoints = new ArrayList<>();
        cejColor = new ArrayList<>();
        cejSize = new ArrayList<>();
        tlaPoints = new ArrayList<>();
        tlaColor = new ArrayList<>();
        tlaSize = new ArrayList<>();
        bonePoints = new ArrayList<>();
        boneColor = new ArrayList<>();
        boneSize = new ArrayList<>();
        teethCejPoints = new HashMap<>();
        tlaPointsByNum = new HashMap<>();
        bonePointsByNum = new HashMap<>();
        bimasks = new HashMap<>();
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

        for (int i = 11; i <= 48; i++) {
            bimasks.put(String.valueOf(i), Mat.zeros(3000, 3000, CvType.CV_8UC1));
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

    public Map<String, Object> parseIniFile(String filepath) throws IOException {
        initialize();
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
        saveMasks();
        drawAndMapCejMask();
        drawAndMapBoneMask();

        return getAnalysisData();
    }

    public Map<Integer, Map<String, Object>> calculateAdjustedCejBoneDistances() {
        Map<Integer, Map<String, Object>> result = new HashMap<>();

        // 상악 치아와 하악 치아 구분
        Set<Integer> maxillaryTeeth = Set.of(11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28);
        Set<Integer> mandibularTeeth = Set.of(31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48);

        // 각 치아의 기준 Y 좌표 계산 (상악: 최대 Y, 하악: 최소 Y)
        Map<Integer, Double> yReferenceByTooth = new HashMap<>();

        for (int i = 0; i < teethPoints.size(); i++) {
            int toothNum = teethNum.get(i);
            List<Point> points = teethPoints.get(i);

            for (Point p : points) {
                if (maxillaryTeeth.contains(toothNum)) {
                    // 상악 치아의 경우 최대 y 값을 기준으로 설정
                    yReferenceByTooth.put(toothNum, Math.max(yReferenceByTooth.getOrDefault(toothNum, Double.MIN_VALUE), p.y));
                } else if (mandibularTeeth.contains(toothNum)) {
                    // 하악 치아의 경우 최소 y 값을 기준으로 설정
                    yReferenceByTooth.put(toothNum, Math.min(yReferenceByTooth.getOrDefault(toothNum, Double.MAX_VALUE), p.y));
                }
            }
        }

        // CEJ 조정 및 거리 계산
        for (Map.Entry<Integer, List<Point>> entry : teethCejPoints.entrySet()) {
            int toothNum = entry.getKey();
            List<Point> cejList = entry.getValue();

            Map<String, Object> toothData = result.computeIfAbsent(toothNum, k -> new HashMap<>());
            List<Map<String, Double>> adjustedCejPoints = new ArrayList<>();
            List<Double> cejDistances = new ArrayList<>();

            if (yReferenceByTooth.containsKey(toothNum)) {
                double yReference = yReferenceByTooth.get(toothNum);

                for (Point cejPoint : cejList) {
                    // 상악 치아는 최대 Y를 기준으로, 하악 치아는 최소 Y를 기준으로 조정
                    double adjustedY = maxillaryTeeth.contains(toothNum) ? yReference - cejPoint.y : cejPoint.y - yReference;
                    adjustedCejPoints.add(Map.of("x", cejPoint.x, "y", adjustedY));
                    cejDistances.add(Math.abs(adjustedY));
                }
            }

            toothData.put("cejPoints", cejList);
            toothData.put("adjustedCejPoints", adjustedCejPoints);
            toothData.put("cejDistances", cejDistances);
        }

        // Bone 조정 및 거리 계산
        for (Map.Entry<Integer, List<Point>> entry : bonePointsByNum.entrySet()) {
            int toothNum = entry.getKey();
            List<Point> boneList = entry.getValue();

            Map<String, Object> toothData = result.computeIfAbsent(toothNum, k -> new HashMap<>());
            List<Map<String, Double>> adjustedBonePoints = new ArrayList<>();
            List<Double> boneDistances = new ArrayList<>();

            if (yReferenceByTooth.containsKey(toothNum)) {
                double yReference = yReferenceByTooth.get(toothNum);

                for (Point bonePoint : boneList) {
                    // 상악 치아는 최대 Y를 기준으로, 하악 치아는 최소 Y를 기준으로 조정
                    double adjustedY = maxillaryTeeth.contains(toothNum) ? yReference - bonePoint.y : bonePoint.y - yReference;
                    adjustedBonePoints.add(Map.of("x", bonePoint.x, "y", adjustedY));
                    boneDistances.add(Math.abs(adjustedY));
                }
            }

            toothData.put("bonePoints", boneList);
            toothData.put("adjustedBonePoints", adjustedBonePoints);
            toothData.put("boneDistances", boneDistances);
        }

        return result;
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
                int minY = toothBoundingBox.y - 50;
                int maxY = toothBoundingBox.y + toothBoundingBox.height + 50;

                for (Point cejPoint : points) {
                    if (toothBoundingBox.contains(cejPoint) &&
                            cejPoint.y >= minY && cejPoint.y <= maxY) {
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
                int minY = toothBoundingBox.y - 50;
                int maxY = toothBoundingBox.y + toothBoundingBox.height + 50;

                for (Point cejPoint : points) {
                    if (toothBoundingBox.contains(cejPoint) &&
                            cejPoint.y >= minY && cejPoint.y <= maxY) {
                        int toothNum = teethNum.get(j);
                        bonePointsByNum.computeIfAbsent(toothNum, k -> new ArrayList<>()).add(cejPoint);
                        Imgproc.circle(boneMappedOnlyMask, cejPoint, 3, new Scalar(0, 255, 0), -1);
                    }
                }
            }
        }
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
