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

    private Map<Integer, List<List<Point>>> tlaPointsByNum;
    private Map<String, Mat> bimasks;
    private Map<Integer, Double> yReferenceByTooth;
    private Map<Integer, RotatedRect> maxBoundingBoxMap = new HashMap<>();
    private Map<Integer, List<Point>> allPointsByTooth = new HashMap<>();

    private Map<Integer, List<Point>> filteredCejPointsByTooth;
    private Map<Integer, List<Point>> filteredBonePointsByTooth;
    private Map<Integer, List<List<Point>>> filteredTlaPointsByTooth = new HashMap<>();

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
        filteredCejPointsByTooth = new HashMap<>();
        tlaPointsByNum = new HashMap<>();
        filteredBonePointsByTooth = new HashMap<>();
        bimasks = new HashMap<>();
        yReferenceByTooth = new HashMap<>();
        maxBoundingBoxMap.clear();  // 바운딩 박스 정보 초기화
        allPointsByTooth.clear();   // 치아 포인트 정보 초기화
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
            } else if (work.equals("S") && line.startsWith("DD")) {
                type_ = "D";
            } else if (work.equals("S") && line.startsWith("RBLD")){
                type_ = "RBL";
            } else if (work.equals("S") && line.startsWith("TRLD")) {
                type_ = "TRL";
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
        drawCombinedMask();
        drawTlaMask(); // TLA Mask 그리기 추가

        Map<Integer, List<Point>> intersectionsByTooth = findAndMarkIntersections();
        printIntersectionsByTooth(intersectionsByTooth);
        // 작은 영역 제거 (최소 면적을 900으로 설정)
        removeIslands(bimasks, 900);

        saveMasks();

        //TODO:- 테스트용 (cej, bone 좌표값 출력확인)
        //printFilteredPoints();
        // 거리값 출력
        printDistancesBetweenMinMaxXCoordinates();

        return getAnalysisData();
    }

    // function: 치아 번호별  cej, bone 교점 거리의 최솟값을 구하는 함수
    // Smaller Distance:- cej, bone 교점 거리의 최솟값
    public void printDistancesBetweenMinMaxXCoordinates() {
        // CEJ Points의 x축 최대 및 최소 좌표 간의 거리 계산 및 출력
        System.out.println("Distances Between Min and Max X Coordinates for CEJ Points (teethCejPoints):");
        Map<Integer, Double> cejDistances = new HashMap<>();

        for (Map.Entry<Integer, List<Point>> entry : filteredCejPointsByTooth.entrySet()) {
            int toothNum = entry.getKey();
            List<Point> points = entry.getValue();

            if (points.isEmpty()) continue;

            double minX = Double.MAX_VALUE;
            double maxX = Double.MIN_VALUE;
            Point minXPoint = null;
            Point maxXPoint = null;

            for (Point point : points) {
                if (point.x < minX) {
                    minX = point.x;
                    minXPoint = point;
                }
                if (point.x > maxX) {
                    maxX = point.x;
                    maxXPoint = point;
                }
            }

            if (minXPoint != null && maxXPoint != null) {
                double distance = Math.sqrt(Math.pow(maxXPoint.x - minXPoint.x, 2) + Math.pow(maxXPoint.y - minXPoint.y, 2));
                cejDistances.put(toothNum, distance);
                System.out.println("Tooth Number: " + toothNum);
                System.out.println("    CEJ Min X Point: " + minXPoint);
                System.out.println("    CEJ Max X Point: " + maxXPoint);
                System.out.println("    CEJ Distance between Min and Max X Points: " + distance);
            }
        }

        // Bone Points의 x축 최대 및 최소 좌표 간의 거리 계산 및 출력
        System.out.println("\nDistances Between Min and Max X Coordinates for Bone Points (bonePointsByNum):");
        Map<Integer, Double> boneDistances = new HashMap<>();

        for (Map.Entry<Integer, List<Point>> entry : filteredBonePointsByTooth.entrySet()) {
            int toothNum = entry.getKey();
            List<Point> points = entry.getValue();

            if (points.isEmpty()) continue;

            double minX = Double.MAX_VALUE;
            double maxX = Double.MIN_VALUE;
            Point minXPoint = null;
            Point maxXPoint = null;

            for (Point point : points) {
                if (point.x < minX) {
                    minX = point.x;
                    minXPoint = point;
                }
                if (point.x > maxX) {
                    maxX = point.x;
                    maxXPoint = point;
                }
            }

            if (minXPoint != null && maxXPoint != null) {
                double distance = Math.sqrt(Math.pow(maxXPoint.x - minXPoint.x, 2) + Math.pow(maxXPoint.y - minXPoint.y, 2));
                boneDistances.put(toothNum, distance);
                System.out.println("Tooth Number: " + toothNum);
                System.out.println("    Bone Min X Point: " + minXPoint);
                System.out.println("    Bone Max X Point: " + maxXPoint);
                System.out.println("    Bone Distance between Min and Max X Points: " + distance);
            }
        }

        // CEJ와 Bone 거리 중 작은 값을 비교하여 출력
        System.out.println("\nSmaller distances between CEJ and Bone for each tooth:");
        for (int toothNum = 11; toothNum <= 48; toothNum++) {
            Double cejDistance = cejDistances.get(toothNum);
            Double boneDistance = boneDistances.get(toothNum);

            if (cejDistance == null && boneDistance == null) {
                System.out.println("Tooth Number: " + toothNum + " - No data available for CEJ or Bone.");
            } else if (cejDistance == null) {
                System.out.println("Tooth Number: " + toothNum + " - Only Bone distance available: " + boneDistance);
            } else if (boneDistance == null) {
                System.out.println("Tooth Number: " + toothNum + " - Only CEJ distance available: " + cejDistance);
            } else {
                double smallerDistance = Math.min(cejDistance, boneDistance);
                System.out.println("Tooth Number: " + toothNum);
                System.out.println("    CEJ Distance: " + cejDistance);
                System.out.println("    Bone Distance: " + boneDistance);
                System.out.println("    Smaller Distance: " + smallerDistance);
            }
        }
    }


    //TLA 각도 찾기
    private Map<Integer, List<Point>> findAndMarkIntersections() {
        Map<Integer, List<Point>> intersectionsByTooth = new HashMap<>();

        // TLA와 CEJ, TLA와 Bone, CEJ와 Bone 교차점을 탐색
        for (Map.Entry<Integer, List<List<Point>>> entry : filteredTlaPointsByTooth.entrySet()) {
            int toothNum = entry.getKey();
            List<List<Point>> tlaSegments = entry.getValue();

            for (List<Point> tlaSegment : tlaSegments) {
                if (tlaSegment.size() >= 2) {
                    // TLA 각도 계산
                    double dx = tlaSegment.get(1).x - tlaSegment.get(0).x;
                    double dy = tlaSegment.get(1).y - tlaSegment.get(0).y;
                    double angleRadians = Math.atan2(dy, dx);
                    double angleDegrees = Math.toDegrees(angleRadians);

                    //TODO:- 테스트용 print (추후 삭제)
                    System.out.println("Tooth " + toothNum + " TLA : " + angleDegrees + " 도");

                    // CEJ와 Bone 교차점 찾기
                    List<Point> cejIntersections = findIntersections(Map.of(toothNum, tlaSegment), filteredCejPointsByTooth, new Scalar(255, 0, 0));
                    List<Point> boneIntersections = findIntersections(Map.of(toothNum, tlaSegment), filteredBonePointsByTooth, new Scalar(0, 255, 255));

                    intersectionsByTooth.computeIfAbsent(toothNum, k -> new ArrayList<>()).addAll(cejIntersections);
                    intersectionsByTooth.computeIfAbsent(toothNum, k -> new ArrayList<>()).addAll(boneIntersections);
                }
            }
        }

        List<Point> cejBoneIntersections = findIntersections(filteredCejPointsByTooth, filteredBonePointsByTooth, new Scalar(0, 0, 255)); // CEJ와 Bone 교차점
        for (Map.Entry<Integer, List<Point>> entry : filteredCejPointsByTooth.entrySet()) {
            int toothNum = entry.getKey();
            intersectionsByTooth.computeIfAbsent(toothNum, k -> new ArrayList<>()).addAll(cejBoneIntersections);
        }

        return intersectionsByTooth;
    }

    // 교차점을 찾는 메서드
    private List<Point> findIntersections(
            Map<Integer, List<Point>> line1Points,
            Map<Integer, List<Point>> line2Points,
            Scalar intersectionColor) {

        List<Point> intersections = new ArrayList<>();

        for (Map.Entry<Integer, List<Point>> entry : line1Points.entrySet()) {
            int toothNum = entry.getKey();
            List<Point> line1 = entry.getValue();
            List<Point> line2 = line2Points.get(toothNum);

            if (line2 == null || line1.size() < 2 || line2.size() < 2) continue;

            for (int i = 0; i < line1.size() - 1; i++) {
                Point p1 = line1.get(i);
                Point p2 = line1.get(i + 1);

                for (int j = 0; j < line2.size() - 1; j++) {
                    Point q1 = line2.get(j);
                    Point q2 = line2.get(j + 1);

                    Point intersection = getIntersection(p1, p2, q1, q2);
                    if (intersection != null) {
                        Imgproc.circle(combinedMask, intersection, 3, intersectionColor, -1);
                        intersections.add(intersection); // 교차점을 리스트에 추가
                    }
                }
            }
        }

        return intersections;
    }


    private static Point getIntersection(Point p1, Point p2, Point q1, Point q2) {
        double a1 = p2.y - p1.y;
        double b1 = p1.x - p2.x;
        double c1 = a1 * p1.x + b1 * p1.y;

        double a2 = q2.y - q1.y;
        double b2 = q1.x - q2.x;
        double c2 = a2 * q1.x + b2 * q1.y;

        double delta = a1 * b2 - a2 * b1;
        if (delta == 0) return null;

        double x = (b2 * c1 - b1 * c2) / delta;
        double y = (a1 * c2 - a2 * c1) / delta;

        if (isBetween(p1, p2, new Point(x, y)) && isBetween(q1, q2, new Point(x, y))) {
            return new Point(x, y);
        } else {
            return null;
        }
    }

    private static boolean isBetween(Point p, Point q, Point r) {
        return r.x >= Math.min(p.x, q.x) && r.x <= Math.max(p.x, q.x)
                && r.y >= Math.min(p.y, q.y) && r.y <= Math.max(p.y, q.y);
    }

    private static void printIntersectionsByTooth(Map<Integer, List<Point>> intersectionsByTooth) {
        for (Map.Entry<Integer, List<Point>> entry : intersectionsByTooth.entrySet()) {
            int toothNum = entry.getKey();
            List<Point> intersections = entry.getValue();

            //TODO:- 테스트용 (추후 삭제) print
            System.out.println("치아 번호: " + toothNum + " - 교차점 좌표:");
            for (Point intersection : intersections) {
                System.out.println("    " + intersection);
            }
        }
    }

    // verticalLength :- 저장된 최대 바운딩 박스의 세로 길이
    private  void drawCombinedMask() {
        // 치아 폴리곤 그리기 및 폴리곤 기반의 최대 바운딩 박스 그리기
        for (int i = 0; i < teethPoints.size(); i++) {
            int toothNum = teethNum.get(i);
            if (toothNum < 11 || toothNum > 48) continue;

            List<Point> points = teethPoints.get(i);
            if (points.size() < 3) continue;

            MatOfPoint pts = new MatOfPoint();
            pts.fromList(points);
            int thickness = teethSize.get(i);

            double toothArea = Imgproc.contourArea(pts);
            if (toothArea < 900) continue;  // 치아 폴리곤의 최소 면적 필터

            // 치아 폴리곤 그리기 - 흰색
            Imgproc.polylines(combinedMask, List.of(pts), true, new Scalar(255, 255, 255), thickness);
            Imgproc.fillPoly(combinedMask, List.of(pts), new Scalar(255, 255, 255));

            // 폴리곤 좌표를 기반으로 최소 외접 직사각형 생성
            MatOfPoint2f pointsMat = new MatOfPoint2f(points.toArray(new Point[0]));
            RotatedRect rotatedBoundingBox = Imgproc.minAreaRect(pointsMat);

            // 최대 면적을 가진 바운딩 박스 저장
            if (maxBoundingBoxMap.containsKey(toothNum)) {
                RotatedRect existingBox = maxBoundingBoxMap.get(toothNum);
                if (existingBox.size.area() < rotatedBoundingBox.size.area()) {
                    maxBoundingBoxMap.put(toothNum, rotatedBoundingBox);
                }
            } else {
                maxBoundingBoxMap.put(toothNum, rotatedBoundingBox);
            }
        }
        // 저장된 최대 바운딩 박스의 세로 길이 출력
        System.out.println("Vertical length of bounding boxes for each tooth:");
        for (Map.Entry<Integer, RotatedRect> entry : maxBoundingBoxMap.entrySet()) {
            RotatedRect maxBox = entry.getValue();
            int toothNum = entry.getKey();

            // 세로 길이 출력
            double verticalLength = maxBox.size.height;
            System.out.println("Tooth Number: " + toothNum);
            System.out.println("    Vertical Length: " + verticalLength);
        }


        // 저장된 최대 바운딩 박스만 그리기
        for (Map.Entry<Integer, RotatedRect> entry : maxBoundingBoxMap.entrySet()) {
            RotatedRect maxBox = entry.getValue();
            int toothNum = entry.getKey();

            Point[] boxPoints = new Point[4];
            maxBox.points(boxPoints);
            for (int j = 0; j < 4; j++) {
                Imgproc.line(combinedMask, boxPoints[j], boxPoints[(j + 1) % 4], new Scalar(0, 255, 255), 2);
            }

            // CEJ와 바운딩 박스 상단 경계 사이의 모든 거리 계산
            List<Point> cejPointsForTooth = filteredCejPointsByTooth.get(toothNum);
            if (cejPointsForTooth != null) {
            }
        }


        // CEJ 폴리곤 그리기 (drawAndMapCejMask의 필터 조건 반영)
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

                List<Point> validCejPoints = new ArrayList<>();
                for (Point cejPoint : points) {
                    if (toothBoundingBox.contains(cejPoint)) {
                        validCejPoints.add(cejPoint);
                    }
                }

                if (validCejPoints.size() >= 2) {
                    MatOfPoint validPts = new MatOfPoint();
                    validPts.fromList(validCejPoints);
                    Imgproc.polylines(combinedMask, Collections.singletonList(validPts), false, new Scalar(0, 0, 255), 2);
                }
            }
        }

        // Bone 폴리곤 그리기 (drawAndMapBoneMask의 필터 조건 반영)
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

                List<Point> validBonePoints = new ArrayList<>();
                for (Point bonePoint : points) {
                    if (toothBoundingBox.contains(bonePoint)) {
                        validBonePoints.add(bonePoint);
                    }
                }

                if (validBonePoints.size() >= 2) {
                    MatOfPoint validPts = new MatOfPoint();
                    validPts.fromList(validBonePoints);
                    Imgproc.polylines(combinedMask, Collections.singletonList(validPts), false, new Scalar(0, 255, 0), 2);
                }
            }
        }

        // TLA 폴리곤 그리기 (drawTlaMask의 필터 조건 반영)
        double maxAllowedDistance = 150.0;
        for (Map.Entry<Integer, List<List<Point>>> entry : tlaPointsByNum.entrySet()) {
            int toothNum = entry.getKey();
            List<Point> filteredToothPoints = allPointsByTooth.get(toothNum);

            if (filteredToothPoints == null || filteredToothPoints.size() < 3) continue;

            MatOfPoint2f toothPoly = new MatOfPoint2f();
            toothPoly.fromArray(filteredToothPoints.toArray(new Point[0]));

            for (List<Point> tlaContour : entry.getValue()) {
                List<Point> filteredTlaPoints = new ArrayList<>();
                for (Point tlaPoint : tlaContour) {
                    double distance = Imgproc.pointPolygonTest(toothPoly, tlaPoint, true);
                    if (Math.abs(distance) <= maxAllowedDistance) {
                        filteredTlaPoints.add(tlaPoint);
                    }
                }

                if (filteredTlaPoints.size() >= 2) {
                    MatOfPoint filteredPts = new MatOfPoint();
                    filteredPts.fromList(filteredTlaPoints);
                    Imgproc.polylines(combinedMask, List.of(filteredPts), true, new Scalar(0, 0, 255), 2);
                }
            }
        }
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
                List<Point> cejList = filteredCejPointsByTooth.getOrDefault(toothNum, Collections.emptyList());
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
                List<Point> boneList = filteredBonePointsByTooth.getOrDefault(toothNum, Collections.emptyList());
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
                    filteredCejPointsByTooth.put(toothNum, validCejPoints); // 필터링된 좌표 저장
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
                    filteredBonePointsByTooth.put(toothNum, validBonePoints); // 필터링된 좌표 저장
                }
            }
        }
    }


    private void drawTlaMask() {
        double maxAllowedDistance = 150.0; // 폴리곤과의 최대 허용 거리 설정

        for (Map.Entry<Integer, List<List<Point>>> entry : tlaPointsByNum.entrySet()) {
            int toothNum = entry.getKey();
            List<Point> filteredToothPoints = allPointsByTooth.get(toothNum); // 필터링된 좌표 맵 사용

            // 필터링된 치아 폴리곤이 없는 경우 건너뛰기
            if (filteredToothPoints == null || filteredToothPoints.size() < 3) continue;

            MatOfPoint2f toothPoly = new MatOfPoint2f();
            toothPoly.fromArray(filteredToothPoints.toArray(new Point[0]));

            List<List<Point>> filteredTlaSegments = new ArrayList<>();

            for (List<Point> tlaContour : entry.getValue()) {
                List<Point> filteredTlaPoints = new ArrayList<>();

                // 각 TLA 좌표에 대해 필터링된 치아 폴리곤과의 거리 계산 후 필터링
                for (Point tlaPoint : tlaContour) {
                    double distance = Imgproc.pointPolygonTest(toothPoly, tlaPoint, true);
                    if (Math.abs(distance) <= maxAllowedDistance) {
                        filteredTlaPoints.add(tlaPoint);
                    }
                }

                // 필터링된 좌표가 2개 이상일 때만 저장
                if (filteredTlaPoints.size() >= 2) {
                    filteredTlaSegments.add(filteredTlaPoints);
                    MatOfPoint filteredPts = new MatOfPoint();
                    filteredPts.fromList(filteredTlaPoints);
                    Imgproc.polylines(tlaMask, List.of(filteredPts), true, new Scalar(0, 0, 255), 2);
                }
            }

            // 필터링된 TLA 좌표를 toothNum에 따라 저장
            if (!filteredTlaSegments.isEmpty()) {
                filteredTlaPointsByTooth.put(toothNum, filteredTlaSegments);
            }
        }
    }

    //TODO:- 테스트용 (추후 삭제)
    public void printFilteredPoints() {
        // Print teethCejPoints
        System.out.println("Filtered CEJ Points (teethCejPoints):");
        for (Map.Entry<Integer, List<Point>> entry : filteredCejPointsByTooth.entrySet()) {
            int toothNum = entry.getKey();
            List<Point> points = entry.getValue();
            System.out.println("Tooth Number: " + toothNum);
            for (Point point : points) {
                System.out.println("    " + point);
            }
        }

        // Print bonePointsByNum
        System.out.println("\nFiltered Bone Points (bonePointsByNum):");
        for (Map.Entry<Integer, List<Point>> entry : filteredBonePointsByTooth.entrySet()) {
            int toothNum = entry.getKey();
            List<Point> points = entry.getValue();
            System.out.println("Tooth Number: " + toothNum);
            for (Point point : points) {
                System.out.println("    " + point);
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
        result.put("teethCejPoints", filteredCejPointsByTooth);
        result.put("tlaPointsByNum", tlaPointsByNum);
        result.put("bonePointsByNum", filteredBonePointsByTooth);
        return result;
    }
}