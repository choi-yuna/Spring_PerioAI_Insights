package com.fas.PrtioAI_Insights.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgcodecs.Imgcodecs;
import org.springframework.stereotype.Service;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.*;

@Service
public class CejBoneDistancesService {
    // 기존 변수 선언
    private final ObjectMapper objectMapper;
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
    private Map<Integer, Double> yReferenceByTooth;
    private Map<Integer, List<Point>> allPointsByTooth = new HashMap<>();

    private Mat combinedMask, cejMask, mappedCejMask, tlaMask, boneMask, cejMappedOnlyMask, boneMappedOnlyMask;

    static {
        try {
            String opencvDll = "/libs/opencv_java4100.dll";
            InputStream in = CejBoneDistancesService.class.getResourceAsStream(opencvDll);
            if (in == null) {
                throw new RuntimeException("DLL 파일을 찾을 수 없습니다: " + opencvDll);
            }
            File tempDll = File.createTempFile("opencv_java4100", ".dll");
            Files.copy(in, tempDll.toPath(), StandardCopyOption.REPLACE_EXISTING);
            System.load(tempDll.getAbsolutePath());
            tempDll.deleteOnExit();
        } catch (IOException e) {
            throw new RuntimeException("OpenCV 라이브러리 로드 실패", e);
        }
    }

    public CejBoneDistancesService(ObjectMapper objectMapper) {
        this.objectMapper = objectMapper;
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
        yReferenceByTooth = new HashMap<>();
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
        Imgcodecs.imwrite("tlaMask.png", tlaMask); // TLA Mask 저장 추가
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

        drawTeethMasks();
        drawAndMapCejMask();
        drawAndMapBoneMask();
        drawTlaMask(); // TLA Mask 그리기 추가

        // 작은 영역 제거 (최소 면적을 900으로 설정)
        removeIslands(bimasks, 900);

        saveMasks();
        return getAnalysisData();
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
    private void drawTeethMasks() {

        for (int i = 0; i < teethPoints.size(); i++) {
            int toothNum = teethNum.get(i);
            if (toothNum < 11 || toothNum > 48) continue;

            List<Point> points = teethPoints.get(i);
            if (points.size() < 3) {
                continue;
            }

            MatOfPoint pts = new MatOfPoint();
            pts.fromList(points);
            int thickness = teethSize.get(i);

            double area = Imgproc.contourArea(pts);
            if (area < 900) {
                continue;
            }

            allPointsByTooth.computeIfAbsent(toothNum, k -> new ArrayList<>()).addAll(points);


            Imgproc.polylines(combinedMask, List.of(pts), true, new Scalar(255, 255, 255), thickness);
            Imgproc.fillPoly(combinedMask, List.of(pts), new Scalar(255, 255, 255));
        }

            for (Map.Entry<Integer, List<Point>> entry : allPointsByTooth.entrySet()) {
                int toothNum = entry.getKey();
                List<Point> combinedPoints = entry.getValue();
                double minY = Double.MAX_VALUE;
                double maxY = Double.MIN_VALUE;

                for (Point p : combinedPoints) {
                    if (p.y < minY) minY = p.y;
                    if (p.y > maxY) maxY = p.y;
                }

                if (toothNum >= 11 && toothNum <= 28) {
                    yReferenceByTooth.put(toothNum, maxY); // 상악 최대 Y 기준
                } else if (toothNum >= 31 && toothNum <= 48) {
                    yReferenceByTooth.put(toothNum, minY); // 하악 최소 Y 기준
                }
            }
    }


    public Set<Integer> filterTeethFromJson(String jsonFilePath) {
        Set<Integer> healthyTeeth = new HashSet<>();
        try {
            JsonNode root = objectMapper.readTree(new File(jsonFilePath));
            JsonNode annotationData = root.path("Annotation_Data");

            if (annotationData.isArray()) {
                annotationData.get(0).fields().forEachRemaining(entry -> {
                    String key = entry.getKey();
                    String value = entry.getValue().asText();
                    if (value.equals("1")) {
                        healthyTeeth.add(Integer.parseInt(key));
                    }
                });
            }
        } catch (IOException e) {
            System.err.println("JSON 파일 처리 중 오류 발생: " + e.getMessage());
        }
        return healthyTeeth;
    }

    public Map<Integer, Map<String, Object>> calculateAdjustedCejBoneDistances(String jsonFilePath) {
        Set<Integer> healthyTeeth = filterTeethFromJson(jsonFilePath);
        Map<Integer, Map<String, Object>> result = new HashMap<>();
        Set<Integer> maxillaryTeeth = Set.of(11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28);

        for (int toothNum = 11; toothNum <= 48; toothNum++) {
            Map<String, Object> toothData = new HashMap<>();

            // 정상 치아 번호라면 좌표 계산
            if (healthyTeeth.contains(toothNum)) {
                List<Point> cejList = teethCejPoints.getOrDefault(toothNum, Collections.emptyList());
                List<Map<String, Double>> adjustedCejPoints = new ArrayList<>();
                List<Double> cejDistances = new ArrayList<>();

                if (yReferenceByTooth.containsKey(toothNum)) {
                    double yReference = yReferenceByTooth.get(toothNum);
                    boolean isMaxillary = maxillaryTeeth.contains(toothNum);

                    for (Point cejPoint : cejList) {
                        double adjustedY = isMaxillary ? yReference - cejPoint.y : cejPoint.y - yReference;
                        adjustedCejPoints.add(Map.of("x", cejPoint.x, "y", adjustedY));
                        cejDistances.add(Math.abs(adjustedY));
                    }
                }

                toothData.put("cejPoints", cejList);
                toothData.put("adjustedCejPoints", adjustedCejPoints);
                toothData.put("cejDistances", cejDistances);
                List<Point> boneList = bonePointsByNum.getOrDefault(toothNum, Collections.emptyList());
                List<Map<String, Double>> adjustedBonePoints = new ArrayList<>();
                List<Double> boneDistances = new ArrayList<>();

                if (yReferenceByTooth.containsKey(toothNum)) {
                    double yReference = yReferenceByTooth.get(toothNum);
                    boolean isMaxillary = maxillaryTeeth.contains(toothNum);

                    for (Point bonePoint : boneList) {
                        double adjustedY = isMaxillary ? yReference - bonePoint.y : bonePoint.y - yReference;
                        adjustedBonePoints.add(Map.of("x", bonePoint.x, "y", adjustedY));
                        boneDistances.add(Math.abs(adjustedY));
                    }
                }

                toothData.put("bonePoints", boneList);
                toothData.put("adjustedBonePoints", adjustedBonePoints);
                toothData.put("boneDistances", boneDistances);

            } else {
                // 비정상 치아 번호는 값을 null로 설정
                toothData.put("cejPoints", null);
                toothData.put("adjustedCejPoints", null);
                toothData.put("cejDistances", null);
                toothData.put("bonePoints", null);
                toothData.put("adjustedBonePoints", null);
                toothData.put("boneDistances", null);
            }

            result.put(toothNum, toothData);
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

            for (Map.Entry<Integer, List<Point>> entry : allPointsByTooth.entrySet()) {
                int toothNum = entry.getKey();
                List<Point> filteredToothPoints = entry.getValue();

                if (filteredToothPoints == null || filteredToothPoints.size() < 3) continue;

                MatOfPoint toothPts = new MatOfPoint();
                toothPts.fromList(filteredToothPoints);

                Rect toothBoundingBox = Imgproc.boundingRect(toothPts);

                int minY = toothBoundingBox.y - 50; // 여유값 추가
                int maxY = toothBoundingBox.y + toothBoundingBox.height + 50;

                // 유효한 CEJ 좌표를 필터링하여 리스트에 저장
                List<Point> validCejPoints = new ArrayList<>();
                for (Point cejPoint : points) {
                    if (toothBoundingBox.contains(cejPoint) &&
                            cejPoint.y >= minY && cejPoint.y <= maxY) { // Y 좌표 필터링 조건 추가
                        validCejPoints.add(cejPoint);
                    }
                }

                // 필터링된 CEJ 좌표가 2개 이상일 때만 폴리라인으로 그리기
                if (validCejPoints.size() >= 2) {
                    MatOfPoint validPts = new MatOfPoint();
                    validPts.fromList(validCejPoints);
                    Imgproc.polylines(cejMappedOnlyMask, Collections.singletonList(validPts), false, new Scalar(0, 255, 0), 2);
                    teethCejPoints.put(toothNum, validCejPoints); // 필터링된 좌표 저장
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

            for (Map.Entry<Integer, List<Point>> entry : allPointsByTooth.entrySet()) {
                int toothNum = entry.getKey();
                List<Point> filteredToothPoints = entry.getValue();

                if (filteredToothPoints == null || filteredToothPoints.size() < 3) continue;

                MatOfPoint toothPts = new MatOfPoint();
                toothPts.fromList(filteredToothPoints);

                Rect toothBoundingBox = Imgproc.boundingRect(toothPts);

                int minY = toothBoundingBox.y - 50; // 여유값 추가
                int maxY = toothBoundingBox.y + toothBoundingBox.height + 50;

                // 유효한 Bone 좌표를 필터링하여 리스트에 저장
                List<Point> validBonePoints = new ArrayList<>();
                for (Point bonePoint : points) {
                    if (toothBoundingBox.contains(bonePoint) &&
                            bonePoint.y >= minY && bonePoint.y <= maxY) { // Y 좌표 필터링 조건 추가
                        validBonePoints.add(bonePoint);
                    }
                }

                // 필터링된 Bone 좌표가 2개 이상일 때만 폴리라인으로 그리기
                if (validBonePoints.size() >= 2) {
                    MatOfPoint validPts = new MatOfPoint();
                    validPts.fromList(validBonePoints);
                    Imgproc.polylines(boneMappedOnlyMask, Collections.singletonList(validPts), false, new Scalar(0, 255, 0), 3);
                    bonePointsByNum.put(toothNum, validBonePoints); // 필터링된 좌표 저장
                }
            }
        }
    }


    private void drawTlaMask() {
        double maxAllowedDistance = 150.0; // 폴리곤과의 최대 허용 거리 설정

        for (Map.Entry<Integer, List<List<Point>>> entry : tlaPointsByNum.entrySet()) {
            int toothNum = entry.getKey();

            // 필터링된 치아 좌표 가져오기
            List<Point> filteredToothPoints = allPointsByTooth.get(toothNum); // 필터링된 좌표 맵 사용

            // 필터링된 치아 폴리곤이 없는 경우 건너뛰기
            if (filteredToothPoints == null || filteredToothPoints.size() < 3) continue;

            // 필터링된 치아 폴리곤 생성
            MatOfPoint2f toothPoly = new MatOfPoint2f();
            toothPoly.fromArray(filteredToothPoints.toArray(new Point[0]));

            for (List<Point> tlaContour : entry.getValue()) {
                List<Point> filteredTlaPoints = new ArrayList<>();

                // 각 TLA 좌표에 대해 필터링된 치아 폴리곤과의 거리 계산 후 필터링
                for (Point tlaPoint : tlaContour) {
                    double distance = Imgproc.pointPolygonTest(toothPoly, tlaPoint, true);
                    if (Math.abs(distance) <= maxAllowedDistance) {
                        filteredTlaPoints.add(tlaPoint);
                    }
                }

                // 필터링된 좌표가 2개 이상일 때만 라인을 그리기
                if (filteredTlaPoints.size() == 2) {
                    MatOfPoint filteredPts = new MatOfPoint();
                    filteredPts.fromList(filteredTlaPoints);
                    Imgproc.polylines(tlaMask, List.of(filteredPts), true, new Scalar(0, 0, 255), 2);

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