package com.fas.PrtioAI_Insights.service;

import com.fas.PrtioAI_Insights.openCV.DicomUtil;
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
import static java.lang.Math.pow;
import static java.lang.Math.sqrt;

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

    // 교점 저장
    private Map<Integer, Map<String, Point>> intersectionsByTooth;
    private Map<Integer, Map<String, List<Point>>> boxIntersections;

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

    private Map<Integer, List<Point>> cejIntersectionsByTooth;
    private Map<Integer, List<Point>> boneIntersectionsByTooth;
    private Map<Integer, Double> tlaAngleByTooth;

    private Map<Integer, List<Point>> toothBoundaries;

    // 각 치아별 4가지 거리 정보를 저장할 전역 맵
    private Map<Integer, Map<String, Double>> distancesByTooth = new HashMap<>();

    // 상악 치아 범위
    private final Set<Integer> MAXILLARY_TEETH = new HashSet<>(Arrays.asList(11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28));


    private Mat combinedMask, cejMask, mappedCejMask, tlaMask, boneMask, cejMappedOnlyMask, boneMappedOnlyMask;

    static {
        try {
            String osName = System.getProperty("os.name").toLowerCase();
            String libraryPath;

            if (osName.contains("win")) {
                // Windows 시스템용 DLL 파일 경로 설정
                libraryPath = "/libs/opencv_java4100.dll";
            } else if (osName.contains("nix") || osName.contains("nux") || osName.contains("mac")) {
                // Linux 또는 Unix 계열 시스템용 SO 파일 경로 설정
                libraryPath = "/libs/libopencv_java4100.so";
            } else {
                throw new UnsupportedOperationException("지원되지 않는 운영체제입니다: " + osName);
            }

            InputStream in = CejBoneDistancesService.class.getResourceAsStream(libraryPath);
            if (in == null) {
                throw new RuntimeException("라이브러리 파일을 찾을 수 없습니다: " + libraryPath);
            }

            File tempLibFile = File.createTempFile("opencv_java", osName.contains("win") ? ".dll" : ".so");
            Files.copy(in, tempLibFile.toPath(), StandardCopyOption.REPLACE_EXISTING);

            System.load(tempLibFile.getAbsolutePath());
            tempLibFile.deleteOnExit(); // 프로그램 종료 시 임시 파일 삭제

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
        cejIntersectionsByTooth = new HashMap<>();
        boneIntersectionsByTooth = new HashMap<>();
        tlaAngleByTooth = new HashMap<>();
        toothBoundaries = new HashMap<>();

        intersectionsByTooth = new HashMap<>();
        boxIntersections = new HashMap<>();
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
            } else if (work.equals("S") && line.startsWith("RBLD")) {
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

        Map<Integer, Map<String, Point>> intersectionsByTooth = findAndMarkLastIntersections();
        printIntersectionsByTooth(intersectionsByTooth);

        // 작은 영역 제거 (최소 면적을 900으로 설정)
        removeIslands(bimasks, 900);

        saveMasks();

        // 필요 시 거리 계산 수행

        return getAnalysisData();
    }




    private Map<Integer, Map<String, Point>> findAndMarkLastIntersections() {
        for (Map.Entry<Integer, List<List<Point>>> entry : filteredTlaPointsByTooth.entrySet()) {
            int toothNum = entry.getKey();
            List<List<Point>> tlaSegments = entry.getValue();

            List<Point> cejIntersections = filteredCejPointsByTooth.get(toothNum);
            List<Point> boneIntersections = filteredBonePointsByTooth.get(toothNum);
            List<Point> toothBoundary = toothBoundaries.get(toothNum);

            if (cejIntersections == null || boneIntersections == null || toothBoundary == null) {
            continue;
            }

            // 바운딩 박스의 중심 Y 좌표 계산
            double boundingBoxCenterY = toothBoundary.stream().mapToDouble(point -> point.y).average().orElse(0);

            for (List<Point> tlaSegment : tlaSegments) {
                if (tlaSegment.size() >= 2) {
                    double dx = tlaSegment.get(1).x - tlaSegment.get(0).x;
                    double dy = tlaSegment.get(1).y - tlaSegment.get(0).y;
                    double length = Math.sqrt(dx * dx + dy * dy);
                    double shiftX = -dy / length;
                    double shiftY = dx / length;

                    List<Point> finalShiftedTlaLeft = new ArrayList<>();
                    List<Point> finalShiftedTlaRight = new ArrayList<>();
                    Point lastCejIntersectionLeft = null;
                    Point lastBoneIntersectionLeft = null;
                    Point lastCejIntersectionRight = null;
                    Point lastBoneIntersectionRight = null;
                    boolean foundLeft = false;
                    boolean foundRight = false;

                    // 기존 TLA를 연장하여 바운딩 박스와의 교점 찾기
                    List<Point> extendedTlaSegment = new ArrayList<>(tlaSegment);
                    Point firstTlaPoint = tlaSegment.get(0);
                    Point lastTlaPoint = tlaSegment.get(tlaSegment.size() - 1);
                    Point extendedFirstTlaPoint = new Point(firstTlaPoint.x - dx * 50 / length, firstTlaPoint.y - dy * 50 / length);
                    Point extendedLastTlaPoint = new Point(lastTlaPoint.x + dx * 50 / length, lastTlaPoint.y + dy * 50 / length);
                    extendedTlaSegment.add(0, extendedFirstTlaPoint);
                    extendedTlaSegment.add(extendedLastTlaPoint);
                    // 기존 TLA와 CEJ 및 치조골 교점 찾기
                    Point cejIntersectionCenter = findClosestIntersection(extendedTlaSegment, cejIntersections);
                    Point boneIntersectionCenter = findClosestIntersection(extendedTlaSegment, boneIntersections);

                    // 기존 TLA와 바운딩 박스 교점 찾기 및 시각화
                    List<Point> originalTlaIntersections = findBoundingBoxIntersections(extendedTlaSegment, toothBoundary);
                    for (Point intersection : originalTlaIntersections) {
                        Imgproc.circle(combinedMask, intersection, 5, new Scalar(255, 0, 255), -1); // 시각화: 기존 TLA 교점
                    }


                    if (cejIntersectionCenter != null) {
                        Imgproc.circle(combinedMask, cejIntersectionCenter, 5, new Scalar(0, 255, 0), -1); // 시각화: 중앙 CEJ 교점
                    }
                    if (boneIntersectionCenter != null) {
                        Imgproc.circle(combinedMask, boneIntersectionCenter, 5, new Scalar(0, 0, 255), -1); // 시각화: 중앙 Bone 교점
                    }

                    // TLA 선을 좌우로 평행 이동 및 연장
                    for (double offset = -50; offset <= 50; offset += 1) {
                        List<Point> shiftedTlaLeft = new ArrayList<>();
                        List<Point> shiftedTlaRight = new ArrayList<>();

                        for (Point p : tlaSegment) {
                            Point leftShifted = new Point(p.x + offset * shiftX, p.y + offset * shiftY);
                            Point rightShifted = new Point(p.x - offset * shiftX, p.y - offset * shiftY);
                            shiftedTlaLeft.add(leftShifted);
                            shiftedTlaRight.add(rightShifted);
                        }

                        // TLA 연장 (양쪽으로 길이 연장)
                        Point firstLeft = shiftedTlaLeft.get(0);
                        Point lastLeft = shiftedTlaLeft.get(shiftedTlaLeft.size() - 1);
                        Point extendedFirstLeft = new Point(firstLeft.x - dx * 50 / length, firstLeft.y - dy * 50 / length);
                        Point extendedLastLeft = new Point(lastLeft.x + dx * 50 / length, lastLeft.y + dy * 50 / length);
                        shiftedTlaLeft.add(0, extendedFirstLeft);
                        shiftedTlaLeft.add(extendedLastLeft);

                        Point firstRight = shiftedTlaRight.get(0);
                        Point lastRight = shiftedTlaRight.get(shiftedTlaRight.size() - 1);
                        Point extendedFirstRight = new Point(firstRight.x - dx * 50 / length, firstRight.y - dy * 50 / length);
                        Point extendedLastRight = new Point(lastRight.x + dx * 50 / length, lastRight.y + dy * 50 / length);
                        shiftedTlaRight.add(0, extendedFirstRight);
                        shiftedTlaRight.add(extendedLastRight);

                        // 교점 찾기
                        if (!shiftedTlaLeft.isEmpty()) {
                            Point currentCejIntersectionLeft = findClosestIntersection(shiftedTlaLeft, cejIntersections);
                            Point currentBoneIntersectionLeft = findClosestIntersection(shiftedTlaLeft, boneIntersections);

                            if (currentCejIntersectionLeft != null && currentBoneIntersectionLeft != null) {
                                lastCejIntersectionLeft = currentCejIntersectionLeft;
                                lastBoneIntersectionLeft = currentBoneIntersectionLeft;
                                finalShiftedTlaLeft = new ArrayList<>(shiftedTlaLeft);
                                foundLeft = true;
                            } else if (foundLeft) break;
                        }

                        if (!shiftedTlaRight.isEmpty()) {
                            Point currentCejIntersectionRight = findClosestIntersection(shiftedTlaRight, cejIntersections);
                            Point currentBoneIntersectionRight = findClosestIntersection(shiftedTlaRight, boneIntersections);

                            if (currentCejIntersectionRight != null && currentBoneIntersectionRight != null) {
                                lastCejIntersectionRight = currentCejIntersectionRight;
                                lastBoneIntersectionRight = currentBoneIntersectionRight;
                                finalShiftedTlaRight = new ArrayList<>(shiftedTlaRight);
                                foundRight = true;
                            } else if (foundRight) break;
                        }
                    }
                    Map<String, Point> toothIntersections = new HashMap<>();
                    if (cejIntersectionCenter != null) {
                        toothIntersections.put("Last_CEJ_Intersection_Center", cejIntersectionCenter);
                    }
                    if (boneIntersectionCenter != null) {
                        toothIntersections.put("Last_Bone_Intersection_Center", boneIntersectionCenter);
                    }

                    if (foundLeft) {
                        toothIntersections.put("Last_CEJ_Intersection_Left", lastCejIntersectionLeft);
                        toothIntersections.put("Last_Bone_Intersection_Left", lastBoneIntersectionLeft);
                        for (int i = 0; i < finalShiftedTlaLeft.size() - 1; i++) {
                            Imgproc.line(combinedMask, finalShiftedTlaLeft.get(i), finalShiftedTlaLeft.get(i + 1), new Scalar(255, 0, 0), 2);
                        }
                        Imgproc.circle(combinedMask, lastCejIntersectionLeft, 5, new Scalar(0, 255, 0), -1);
                        Imgproc.circle(combinedMask, lastBoneIntersectionLeft, 5, new Scalar(0, 0, 255), -1);
                    }

                    if (foundRight) {
                        toothIntersections.put("Last_CEJ_Intersection_Right", lastCejIntersectionRight);
                        toothIntersections.put("Last_Bone_Intersection_Right", lastBoneIntersectionRight);
                        for (int i = 0; i < finalShiftedTlaRight.size() - 1; i++) {
                            Imgproc.line(combinedMask, finalShiftedTlaRight.get(i), finalShiftedTlaRight.get(i + 1), new Scalar(255, 0, 0), 2);
                        }
                        Imgproc.circle(combinedMask, lastCejIntersectionRight, 5, new Scalar(0, 255, 0), -1);
                        Imgproc.circle(combinedMask, lastBoneIntersectionRight, 5, new Scalar(0, 0, 255), -1);
                    }

                    // 바운딩 박스와 TLA 교점 찾기 및 combinedMask 표시
                    List<Point> boundingBoxIntersectionsLeft = findBoundingBoxIntersections(finalShiftedTlaLeft, toothBoundary);
                    List<Point> boundingBoxIntersectionsRight = findBoundingBoxIntersections(finalShiftedTlaRight, toothBoundary);

                    // 상하 교점 분류 및 boxIntersections에 추가
                    Map<String, List<Point>> currentBoxIntersections = boxIntersections.computeIfAbsent(toothNum, k -> new HashMap<>());
                    for (Point intersection : boundingBoxIntersectionsLeft) {
                        String key = intersection.y < boundingBoxCenterY ? "박스 상단 교점" : "박스 하단 교점";
                        currentBoxIntersections.computeIfAbsent(key, k -> new ArrayList<>()).add(intersection);
                        Imgproc.circle(combinedMask, intersection, 5, new Scalar(0, 255, 255), -1); // 시각화: 상단/하단 교점 표시
                    }

                    for (Point intersection : originalTlaIntersections) {
                        String key = intersection.y < boundingBoxCenterY ? "박스 상단 교점" : "박스 하단 교점";
                        currentBoxIntersections.computeIfAbsent(key, k -> new ArrayList<>()).add(intersection);
                    }
                    for (Point intersection : boundingBoxIntersectionsRight) {
                        String key = intersection.y < boundingBoxCenterY ? "박스 상단 교점" : "박스 하단 교점";
                        currentBoxIntersections.computeIfAbsent(key, k -> new ArrayList<>()).add(intersection);
                        Imgproc.circle(combinedMask, intersection, 5, new Scalar(0, 255, 255), -1); // 시각화: 상단/하단 교점 표시
                    }

                    if (!toothIntersections.isEmpty()) {
                        intersectionsByTooth.put(toothNum, toothIntersections);
                    }
                }
            }
        }

        // boxIntersections 출력
//        System.out.println("바운딩 박스 교점 정보:");
//        for (Map.Entry<Integer, Map<String, List<Point>>> entry : boxIntersections.entrySet()) {
//            int toothNum = entry.getKey();
//            Map<String, List<Point>> intersections = entry.getValue();
//            System.out.println("치아 번호: " + toothNum);
//            for (Map.Entry<String, List<Point>> intersectionEntry : intersections.entrySet()) {
//                String position = intersectionEntry.getKey();
//                List<Point> points = intersectionEntry.getValue();
//                System.out.println(" - " + position + ": " + points);
//            }
//        }

        return intersectionsByTooth;
    }

    // 바운딩 박스와 TLA 선 간 교점을 찾는 메서드 (사각형의 윗변과 아랫변에서 발생하는 교점만 반환)
    private List<Point> findBoundingBoxIntersections(List<Point> shiftedTla, List<Point> boundingBox) {
        List<Point> intersections = new ArrayList<>();

        // boundingBox의 네 점을 이용해 사각형의 네 변을 정의합니다.
        Point p1 = boundingBox.get(0);
        Point p2 = boundingBox.get(1);
        Point p3 = boundingBox.get(2);
        Point p4 = boundingBox.get(3);

        // 사각형의 네 변을 정의
        Point[][] edges = {
                {p1, p2}, // 첫 번째 변
                {p2, p3}, // 두 번째 변
                {p3, p4}, // 세 번째 변
                {p4, p1}  // 네 번째 변
        };

        // 네 변 중 가장 위쪽에 있는 변과 가장 아래쪽에 있는 변을 찾기 위해 각 변의 중심 y좌표를 구합니다.
        double[] edgeCenterY = {
                (p1.y + p2.y) / 2,
                (p2.y + p3.y) / 2,
                (p3.y + p4.y) / 2,
                (p4.y + p1.y) / 2
        };

        // 가장 위쪽에 있는 변과 가장 아래쪽에 있는 변을 찾습니다.
        int topEdgeIndex = 0;
        int bottomEdgeIndex = 0;
        for (int i = 1; i < edgeCenterY.length; i++) {
            if (edgeCenterY[i] < edgeCenterY[topEdgeIndex]) {
                topEdgeIndex = i;
            }
            if (edgeCenterY[i] > edgeCenterY[bottomEdgeIndex]) {
                bottomEdgeIndex = i;
            }
        }

        // 윗변과 아랫변 선택
        Point[] topEdge = edges[topEdgeIndex];
        Point[] bottomEdge = edges[bottomEdgeIndex];

        // shiftedTla 선분의 각 점들에 대해 윗변과 아랫변과의 교점만 검사
        for (int i = 0; i < shiftedTla.size() - 1; i++) {
            Point tlaPoint1 = shiftedTla.get(i);
            Point tlaPoint2 = shiftedTla.get(i + 1);

            // 윗변과의 교점 찾기
            Point topIntersection = findExactIntersection(tlaPoint1, tlaPoint2, topEdge[0], topEdge[1]);
            if (topIntersection != null) {
                intersections.add(topIntersection);
            }

            // 아랫변과의 교점 찾기
            Point bottomIntersection = findExactIntersection(tlaPoint1, tlaPoint2, bottomEdge[0], bottomEdge[1]);
            if (bottomIntersection != null) {
                intersections.add(bottomIntersection);
            }
        }

        return intersections;
    }

    private Point findExactIntersection(Point p1, Point p2, Point q1, Point q2) {
        double a1 = p2.y - p1.y;
        double b1 = p1.x - p2.x;
        double c1 = a1 * p1.x + b1 * p1.y;

        double a2 = q2.y - q1.y;
        double b2 = q1.x - q2.x;
        double c2 = a2 * q1.x + b2 * q1.y;

        double delta = a1 * b2 - a2 * b1;

        double x = (b2 * c1 - b1 * c2) / delta;
        double y = (a1 * c2 - a2 * c1) / delta;
        Point intersection = new Point(x, y);

        // 교점이 각 선분의 범위 내에 있는지 확인
        if (isBetween(p1, p2, intersection) && isBetween(q1, q2, intersection)) {
            return intersection;
        } else {
            return null;
        }
    }

    private Point findClosestIntersection(List<Point> tla, List<Point> otherLine) {
        Point closestIntersection = null;
        double minDist = Double.MAX_VALUE;

        for (int i = 0; i < tla.size() - 1; i++) {
            Point p1 = tla.get(i);
            Point p2 = tla.get(i + 1);

            for (int j = 0; j < otherLine.size() - 1; j++) {
                Point q1 = otherLine.get(j);
                Point q2 = otherLine.get(j + 1);

                Point intersection = findExactIntersection(p1, p2, q1, q2);
                if (intersection != null) {
                    double dist = Math.hypot(p1.x - intersection.x, p1.y - intersection.y);
                    if (dist < minDist) {
                        minDist = dist;
                        closestIntersection = intersection;
                    }
                }
            }
        }

        return closestIntersection;
    }



    // 두 선분 간 교차점을 찾는 메서드
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

    // 두 점 간의 교차 여부를 확인하는 메서드
    private static boolean isBetween(Point p, Point q, Point r) {
        double epsilon = 1e-6; // 허용 오차를 조금 더 크게 설정
        return r.x >= Math.min(p.x, q.x) - epsilon && r.x <= Math.max(p.x, q.x) + epsilon
                && r.y >= Math.min(p.y, q.y) - epsilon && r.y <= Math.max(p.y, q.y) + epsilon;
    }



    private static void printIntersectionsByTooth(Map<Integer, Map<String, Point>> intersectionsByTooth) {
        for (Map.Entry<Integer, Map<String, Point>> entry : intersectionsByTooth.entrySet()) {
            int toothNum = entry.getKey();
            Map<String, Point> intersections = entry.getValue();

            //TODO:- 테스트용 (추후 삭제) print
//            System.out.println("치아 번호: " + toothNum + " - 교차점 좌표:");
//            for (Map.Entry<String, Point> intersectionEntry : intersections.entrySet()) {
//                System.out.println("    " + intersectionEntry.getKey() + ": " + intersectionEntry.getValue());
//            }
        }
    }


    // verticalLength :- 저장된 최대 바운딩 박스의 세로 길이
    private void drawCombinedMask() {
        // 치아 폴리곤 그리기 및 회전된 바운딩 박스 그리기
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

            // 회전된 바운딩 박스 생성
            MatOfPoint2f pointsMat = new MatOfPoint2f(points.toArray(new Point[0]));
            RotatedRect rotatedBoundingBox = Imgproc.minAreaRect(pointsMat);

            // TLA 각도를 적용하여 회전시키기
            if (tlaAngleByTooth.containsKey(toothNum)) {
                double tlaAngle = tlaAngleByTooth.get(toothNum);
                rotatedBoundingBox = new RotatedRect(rotatedBoundingBox.center, rotatedBoundingBox.size, tlaAngle);
                //System.out.println("Tooth " + toothNum + " - Applied TLA Angle: " + tlaAngle);  // 디버깅 출력
            }

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

        // 저장된 회전된 바운딩 박스 그리기
        for (Map.Entry<Integer, RotatedRect> entry : maxBoundingBoxMap.entrySet()) {
            RotatedRect maxBox = entry.getValue();
            int toothNum = entry.getKey();

            Point[] boxPoints = new Point[4];
            maxBox.points(boxPoints);
            toothBoundaries.put(toothNum, Arrays.asList(boxPoints));

            // 바운딩 박스의 네 모서리를 연결하여 그리기
            for (int j = 0; j < 4; j++) {
                Imgproc.line(combinedMask, boxPoints[j], boxPoints[(j + 1) % 4], new Scalar(0, 255, 255), 2);
            }

            // 회전 중심을 시각적으로 표시 (확인 용도)
            Imgproc.circle(combinedMask, maxBox.center, 5, new Scalar(255, 0, 0), -1);

            // 바운딩 박스의 회전 각도 출력 (디버깅 용도)
//            System.out.println("Tooth Number: " + toothNum);
//            System.out.println("    Bounding Box Angle: " + maxBox.angle);
//            System.out.println("    Bounding Box Center: " + maxBox.center);
//            System.out.println("    Bounding Box Size: " + maxBox.size);
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



    private void drawAndMapCejMask() {
        // CEJ 교차점 저장을 위한 Map 초기화
        cejIntersectionsByTooth.clear();
        filteredCejPointsByTooth.clear();

        for (int i = 0; i < cejPoints.size(); i++) {
            List<Point> points = cejPoints.get(i);
            if (points.size() < 3) continue;

            // CEJ 색상 필터링 추가
            Scalar cejColors = cejColor.get(i);
            if (cejColors.equals(new Scalar(1, 1, 255, 255))) {  // 흰색 선
                MatOfPoint pts = new MatOfPoint();
                pts.fromList(points);
                int thickness = cejSize.get(i);

                double area = Imgproc.contourArea(pts);
                if (area < 300 || thickness > 2) continue;

                for (int j = 0; j < teethPoints.size(); j++) {
                    int toothNum = teethNum.get(j);
                    if (toothNum < 11 || toothNum > 48) continue;

                    List<Point> toothPoints = teethPoints.get(j);
                    if (toothPoints.size() < 3) continue;

                    MatOfPoint toothPts = new MatOfPoint();
                    toothPts.fromList(toothPoints);

                    double toothArea = Imgproc.contourArea(toothPts);
                    if (toothArea < 900) continue;

                    // CEJ와 치아 폴리곤의 교차점 찾기
                    List<Point> intersections = findIntersectionsBetweenCEJAndTooth(points, toothPoints);
                    List<Point> filteredCejPoints = new ArrayList<>();

                    if (!intersections.isEmpty()) {
                        // 교차점 중 가장 먼저 맞닿는 부분과 마지막 맞닿는 부분 찾기
                        int firstIntersectionIndex = points.indexOf(intersections.get(0));
                        int lastIntersectionIndex = points.indexOf(intersections.get(intersections.size() - 1));

                        // 인덱스가 유효한지 확인
                        if (firstIntersectionIndex >= 0 && lastIntersectionIndex >= firstIntersectionIndex) {
                            filteredCejPoints.addAll(points.subList(firstIntersectionIndex, lastIntersectionIndex + 1));
                        }
                    }

                    // 교차점이 없거나 추가적으로 내부에 포함된 점을 확인하여 추가
                    for (Point cejPoint : points) {
                        if (Imgproc.pointPolygonTest(new MatOfPoint2f(toothPoints.toArray(new Point[0])), cejPoint, false) >= 0) {
                            if (!filteredCejPoints.contains(cejPoint)) {
                                filteredCejPoints.add(cejPoint);
                            }
                        }
                    }

                    if (!filteredCejPoints.isEmpty()) {
                        filteredCejPointsByTooth.put(toothNum, filteredCejPoints);
                        // CEJ 라인 그리기 - 흰색 선
                        MatOfPoint cejPts = new MatOfPoint();
                        cejPts.fromList(filteredCejPoints);
                        Imgproc.polylines(cejMappedOnlyMask, List.of(cejPts), false, new Scalar(255, 255, 255), thickness);
                    }
                }
            } else if (cejColor.equals(new Scalar(0, 0, 0, 0))) {  // 검은색 선
                // 필요 시 검은색 선에 대한 로직 추가
            }
        }
    }


    private void drawAndMapBoneMask() {
        // Bone 교차점 저장을 위한 Map 초기화
        boneIntersectionsByTooth.clear();
        filteredBonePointsByTooth.clear();

        for (int i = 0; i < bonePoints.size(); i++) {
            List<Point> points = bonePoints.get(i);
            if (points.size() < 3) continue;

            MatOfPoint pts = new MatOfPoint();
            pts.fromList(points);
            int thickness = boneSize.get(i);

            double area = Imgproc.contourArea(pts);
            if (area < 200 || thickness > 3) continue;

            for (Map.Entry<Integer, List<Point>> entry : allPointsByTooth.entrySet()) {
                int toothNum = entry.getKey();
                List<Point> filteredToothPoints = entry.getValue();

                if (filteredToothPoints == null || filteredToothPoints.size() < 3) continue;

                // Bone과 치아 폴리곤의 교차점 찾기
                List<Point> intersections = findIntersectionsBetweenBoneAndTooth(points, filteredToothPoints);
                if (!intersections.isEmpty()) {
                    // 가장 작은 X와 큰 X를 기준으로 유효한 Bone 좌표를 필터링
                    Point minXPoint = intersections.stream().min(Comparator.comparingDouble(p -> p.x)).orElse(null);
                    Point maxXPoint = intersections.stream().max(Comparator.comparingDouble(p -> p.x)).orElse(null);

                    List<Point> filteredBonePoints = new ArrayList<>();
                    for (Point bonePoint : points) {
                        if (minXPoint != null && maxXPoint != null && bonePoint.x >= minXPoint.x && bonePoint.x <= maxXPoint.x) {
                            filteredBonePoints.add(bonePoint);
                        }
                    }

                    if (!filteredBonePoints.isEmpty()) {
                        filteredBonePointsByTooth.put(toothNum, filteredBonePoints);
                    }

                    // Bone 라인 그리기 - 녹색
                    MatOfPoint bonePts = new MatOfPoint();
                    bonePts.fromList(filteredBonePoints);
                    Imgproc.polylines(cejMappedOnlyMask, List.of(bonePts), false, new Scalar(0, 255, 0), thickness);
                }
            }
        }
    }




    // CEJ와 치아 폴리곤 간 교차점을 찾는 메서드
    private List<Point> findIntersectionsBetweenCEJAndTooth(List<Point> cejPoints, List<Point> toothPolygon) {
        List<Point> intersections = new ArrayList<>();

        for (int i = 0; i < cejPoints.size() - 1; i++) {
            Point p1 = cejPoints.get(i);
            Point p2 = cejPoints.get(i + 1);

            for (int j = 0; j < toothPolygon.size() - 1; j++) {
                Point q1 = toothPolygon.get(j);
                Point q2 = toothPolygon.get(j + 1);

                Point intersection = getIntersection(p1, p2, q1, q2);
                if (intersection != null) {
                    intersections.add(intersection);
                }
            }
        }
        return intersections;
    }


    // Bone과 치아 폴리곤 간 교차점을 찾는 메서드
    private List<Point> findIntersectionsBetweenBoneAndTooth(List<Point> bonePoints, List<Point> toothPolygon) {
        List<Point> intersections = new ArrayList<>();

        for (int i = 0; i < bonePoints.size() - 1; i++) {
            Point p1 = bonePoints.get(i);
            Point p2 = bonePoints.get(i + 1);

            for (int j = 0; j < toothPolygon.size() - 1; j++) {
                Point q1 = toothPolygon.get(j);
                Point q2 = toothPolygon.get(j + 1);

                Point intersection = getIntersection(p1, p2, q1, q2);
                if (intersection != null) {
                    intersections.add(intersection);
                }
            }
        }
        return intersections;
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


    // 거리 계산 함수
    public Map<Integer, Map<String, List<Double>>> calculateDistances(String jsonFilePath,String dicomFilePath) throws IOException {
        Set<Integer> healthyTeeth = filterTeethFromJson(jsonFilePath);
        Map<Integer, Map<String, List<Double>>> result = new HashMap<>();
        Point previousCejPoint = null;
        Point previousBonePoint = null;

        double[] pixelSpacing = DicomUtil.getPixelSpacing(dicomFilePath);
        int[] dicomDimensions = DicomUtil.getDicomDimensions(dicomFilePath);
        int dicomWidth = dicomDimensions[0];
        int dicomHeight = dicomDimensions[1];
        int imageWidth = 3000;
        int imageHeight = 3000;


        for (Map.Entry<Integer, Map<String, Point>> entry : intersectionsByTooth.entrySet()) {
            int toothNum = entry.getKey();
            Map<String, Point> toothIntersections = entry.getValue();
            Map<String, List<Point>> toothBoxIntersections = boxIntersections.get(toothNum);

            // 상악과 하악을 구분하여 박스 교점 기준 설정
            String keyForBoxIntersections = MAXILLARY_TEETH.contains(toothNum) ? "박스 하단 교점" : "박스 상단 교점";

            // 교점 정보가 없으면 건너뜀
            if (toothBoxIntersections == null || !toothBoxIntersections.containsKey(keyForBoxIntersections)) continue;

            List<Point> boxPoints = toothBoxIntersections.get(keyForBoxIntersections);

            // 좌우 계산을 위해 최소 한 개의 박스 교점이 필요
            if (boxPoints.isEmpty()) continue;

            double leftCejDistance, leftBoneDistance, rightCejDistance, rightBoneDistance, centerCejDistance, centerBoneDistance;
            if (boxPoints.size() < 3 && boxPoints.size() != 0) {
                Point singleBoxPoint = boxPoints.get(0);

                leftCejDistance = DicomUtil.calculatePhysicalDistance(singleBoxPoint, toothIntersections.get("Last_CEJ_Intersection_Left"), pixelSpacing, dicomWidth, dicomHeight, imageWidth, imageHeight,previousCejPoint);
                previousCejPoint = (toothIntersections.get("Last_CEJ_Intersection_Left") != null) ? toothIntersections.get("Last_CEJ_Intersection_Left") : previousCejPoint;

                leftBoneDistance = DicomUtil.calculatePhysicalDistance(singleBoxPoint, toothIntersections.get("Last_Bone_Intersection_Left"), pixelSpacing, dicomWidth, dicomHeight, imageWidth, imageHeight,previousBonePoint);
                previousBonePoint = (toothIntersections.get("Last_Bone_Intersection_Left") != null) ? toothIntersections.get("Last_Bone_Intersection_Left") : previousBonePoint;

                centerCejDistance = DicomUtil.calculatePhysicalDistance(singleBoxPoint, toothIntersections.get("Last_CEJ_Intersection_Center"), pixelSpacing, dicomWidth, dicomHeight, imageWidth, imageHeight,previousCejPoint);
                previousCejPoint = (toothIntersections.get("Last_CEJ_Intersection_Center") != null) ? toothIntersections.get("Last_CEJ_Intersection_Center") : previousCejPoint;

                centerBoneDistance = DicomUtil.calculatePhysicalDistance(singleBoxPoint, toothIntersections.get("Last_Bone_Intersection_Center"), pixelSpacing, dicomWidth, dicomHeight, imageWidth, imageHeight,previousBonePoint);
                previousBonePoint = (toothIntersections.get("Last_Bone_Intersection_Center") != null) ? toothIntersections.get("Last_Bone_Intersection_Center") : previousBonePoint;

                rightCejDistance = DicomUtil.calculatePhysicalDistance(singleBoxPoint, toothIntersections.get("Last_CEJ_Intersection_Right"), pixelSpacing, dicomWidth, dicomHeight, imageWidth, imageHeight,previousCejPoint);
                previousCejPoint = (toothIntersections.get("Last_CEJ_Intersection_Right") != null) ? toothIntersections.get("Last_CEJ_Intersection_Right") : previousCejPoint;

                rightBoneDistance = DicomUtil.calculatePhysicalDistance(singleBoxPoint, toothIntersections.get("Last_Bone_Intersection_Right"), pixelSpacing, dicomWidth, dicomHeight, imageWidth, imageHeight,previousBonePoint);
                previousBonePoint = (toothIntersections.get("Last_Bone_Intersection_Right") != null) ? toothIntersections.get("Last_Bone_Intersection_Right") : previousBonePoint;
            } else {
                // 박스 교점이 두 개 이상일 경우, 좌우 교점으로 거리 계산
                leftCejDistance = DicomUtil.calculatePhysicalDistance(boxPoints.get(0), toothIntersections.get("Last_CEJ_Intersection_Left"), pixelSpacing, dicomWidth, dicomHeight, imageWidth, imageHeight,previousCejPoint);
                previousCejPoint = (toothIntersections.get("Last_CEJ_Intersection_Left") != null) ? toothIntersections.get("Last_CEJ_Intersection_Left") : previousCejPoint;

                leftBoneDistance = DicomUtil.calculatePhysicalDistance(boxPoints.get(0), toothIntersections.get("Last_Bone_Intersection_Left"), pixelSpacing, dicomWidth, dicomHeight, imageWidth, imageHeight,previousBonePoint);
                previousBonePoint = (toothIntersections.get("Last_Bone_Intersection_Left") != null) ? toothIntersections.get("Last_Bone_Intersection_Left") : previousBonePoint;

                centerCejDistance = DicomUtil.calculatePhysicalDistance(boxPoints.get(1), toothIntersections.get("Last_CEJ_Intersection_Center"), pixelSpacing, dicomWidth, dicomHeight, imageWidth, imageHeight,previousCejPoint);
                previousCejPoint = (toothIntersections.get("Last_CEJ_Intersection_Center") != null) ? toothIntersections.get("Last_CEJ_Intersection_Center") : previousCejPoint;

                centerBoneDistance = DicomUtil.calculatePhysicalDistance(boxPoints.get(1), toothIntersections.get("Last_Bone_Intersection_Center"), pixelSpacing, dicomWidth, dicomHeight, imageWidth, imageHeight,previousBonePoint);
                previousBonePoint = (toothIntersections.get("Last_Bone_Intersection_Center") != null) ? toothIntersections.get("Last_Bone_Intersection_Center") : previousBonePoint;

                rightCejDistance = DicomUtil.calculatePhysicalDistance(boxPoints.get(2), toothIntersections.get("Last_CEJ_Intersection_Right"), pixelSpacing, dicomWidth, dicomHeight, imageWidth, imageHeight,previousCejPoint);
                previousCejPoint = (toothIntersections.get("Last_CEJ_Intersection_Right") != null) ? toothIntersections.get("Last_CEJ_Intersection_Right") : previousCejPoint;

                rightBoneDistance = DicomUtil.calculatePhysicalDistance(boxPoints.get(2), toothIntersections.get("Last_Bone_Intersection_Right"), pixelSpacing, dicomWidth, dicomHeight, imageWidth, imageHeight,previousBonePoint);
                previousBonePoint = (toothIntersections.get("Last_Bone_Intersection_Right") != null) ? toothIntersections.get("Last_Bone_Intersection_Right") : previousBonePoint;
            }

            // 결과 맵에 거리 값 저장
            Map<String, List<Double>> distances = new HashMap<>();
            distances.put("adjustedCejPoints", Arrays.asList(leftCejDistance, centerCejDistance, rightCejDistance));
            distances.put("adjustedBonePoints", Arrays.asList(leftBoneDistance, centerBoneDistance, rightBoneDistance));

            // 정상 치아인지 확인하고 결과에 추가
            if (healthyTeeth.contains(toothNum)) {
                result.put(toothNum, distances);
            } else {
                // 비정상 치아 번호인 경우 null 값으로 설정
                Map<String, List<Double>> nullData = new HashMap<>();
                nullData.put("cej", Arrays.asList(null, null, null));
                nullData.put("bone", Arrays.asList(null, null, null));
                result.put(toothNum, nullData);
            }
        }

        return result;
    }

    private double calculatePointDistanceWithFallback(Point p1, Point p2, Point fallbackPoint) {
        if (p1 == null || (p2 == null && fallbackPoint == null)) {
            return Double.MAX_VALUE; // 거리 계산 불가한 경우 무한대 리턴
        }
        Point actualPoint = (p2 != null) ? p2 : fallbackPoint;
        return sqrt(pow(p1.x - actualPoint.x, 2) + pow(p1.y - actualPoint.y, 2));
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