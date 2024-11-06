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

    private Map<Integer, List<Point>> cejIntersectionsByTooth;
    private Map<Integer, List<Point>> boneIntersectionsByTooth;
    private Map<Integer, Double> tlaAngleByTooth;

    private Map<Integer, List<Point>> toothBoundaries;
    // 치아 번호별 바운딩 박스와 TLA 선 간 교차점을 저장하는 변수
    private Map<Integer, List<Point>> boundingBoxIntersectionsByTooth = new HashMap<>();



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
        cejIntersectionsByTooth = new HashMap<>();
        boneIntersectionsByTooth = new HashMap<>();
        tlaAngleByTooth = new HashMap<>();
        toothBoundaries = new HashMap<>();
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



        return getAnalysisData();
    }




    private Map<Integer, Map<String, Point>> findAndMarkLastIntersections() {
        Map<Integer, Map<String, Point>> intersectionsByTooth = new HashMap<>();

        for (Map.Entry<Integer, List<List<Point>>> entry : tlaPointsByNum.entrySet()) {
            int toothNum = entry.getKey();
            List<List<Point>> tlaSegments = entry.getValue();

            List<Point> cejIntersections = cejIntersectionsByTooth.get(toothNum);
            List<Point> boneIntersections = boneIntersectionsByTooth.get(toothNum);
            List<Point> toothBoundary = toothBoundaries.get(toothNum); // 바운딩 박스 가져오기

            if (cejIntersections == null || boneIntersections == null || toothBoundary == null) continue;

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

                    // TLA 선을 좌우로 평행 이동 및 연장
                    for (double offset = -50; offset <= 50; offset += 0.5) {
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

                    // 바운딩 박스와 TLA 교점 찾기

                    List<Point> boundingBoxIntersectionsLeft = findBoundingBoxIntersections(finalShiftedTlaLeft, toothBoundary, toothNum);
                    List<Point> boundingBoxIntersectionsRight = findBoundingBoxIntersections(finalShiftedTlaRight, toothBoundary, toothNum);


                    for (Point intersection : boundingBoxIntersectionsLeft) {
                        Imgproc.circle(combinedMask, intersection, 5, new Scalar(255, 255, 0), -1); // 노란색으로 왼쪽 교점 표시
                    }
                    for (Point intersection : boundingBoxIntersectionsRight) {
                        Imgproc.circle(combinedMask, intersection, 5, new Scalar(255, 255, 0), -1); // 노란색으로 오른쪽 교점 표시
                    }

                    if (!toothIntersections.isEmpty()) {
                        intersectionsByTooth.put(toothNum, toothIntersections);
                    }
                }
            }
        }
        return intersectionsByTooth;
    }


    // 바운딩 박스와 TLA 선 간 교점을 찾는 메서드 (치아 번호별로 교점 좌표를 반환)
    private List<Point> findBoundingBoxIntersections(List<Point> shiftedTla, List<Point> boundingBox, int toothNum) {
        List<Point> intersections = new ArrayList<>();
        for (int i = 0; i < shiftedTla.size() - 1; i++) {
            Point p1 = shiftedTla.get(i);
            Point p2 = shiftedTla.get(i + 1);
            for (int j = 0; j < boundingBox.size(); j++) {
                Point q1 = boundingBox.get(j);
                Point q2 = boundingBox.get((j + 1) % boundingBox.size());
                Point intersection = findExactIntersection(p1, p2, q1, q2);
                if (intersection != null) intersections.add(intersection);
            }
        }

        if (!intersections.isEmpty()) {
            boundingBoxIntersectionsByTooth.put(toothNum, intersections);
            printBoundingBoxIntersections(toothNum, intersections);
        }

        return intersections; // 교차점 리스트 반환
    }


    // 교차점 출력 메서드
    private void printBoundingBoxIntersections(int toothNum, List<Point> intersections) {
        System.out.println("치아 번호: " + toothNum + " - TLA와 Bounding Box 교차점 좌표:");
        for (int i = 0; i < intersections.size(); i++) {
            Point intersection = intersections.get(i);
            System.out.printf("    Intersection_%d: {%.2f, %.2f}%n", i + 1, intersection.x, intersection.y);
        }
    }



    private Point findExactIntersection(Point p1, Point p2, Point q1, Point q2) {
        double a1 = p2.y - p1.y;
        double b1 = p1.x - p2.x;
        double c1 = a1 * p1.x + b1 * p1.y;

        double a2 = q2.y - q1.y;
        double b2 = q1.x - q2.x;
        double c2 = a2 * q1.x + b2 * q1.y;

        double delta = a1 * b2 - a2 * b1;
        if (Math.abs(delta) < 1e-6) return null; // 두 선이 평행할 경우 교점 없음

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


    // TLA가 치아 경계를 벗어났는지 확인하는 메서드
    private boolean isWithinToothBoundary(List<Point> tla, List<Point> toothBoundary) {
        for (Point p : tla) {
            if (Imgproc.pointPolygonTest(new MatOfPoint2f(toothBoundary.toArray(new Point[0])), p, false) < 0) {
                return false;
            }
        }
        return true;
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
        return r.x >= Math.min(p.x, q.x) && r.x <= Math.max(p.x, q.x)
                && r.y >= Math.min(p.y, q.y) && r.y <= Math.max(p.y, q.y);
    }


    private static void printIntersectionsByTooth(Map<Integer, Map<String, Point>> intersectionsByTooth) {
        for (Map.Entry<Integer, Map<String, Point>> entry : intersectionsByTooth.entrySet()) {
            int toothNum = entry.getKey();
            Map<String, Point> intersections = entry.getValue();

            //TODO:- 테스트용 (추후 삭제) print
            System.out.println("치아 번호: " + toothNum + " - 교차점 좌표:");
            for (Map.Entry<String, Point> intersectionEntry : intersections.entrySet()) {
                System.out.println("    " + intersectionEntry.getKey() + ": " + intersectionEntry.getValue());
            }
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
                System.out.println("Tooth " + toothNum + " - Applied TLA Angle: " + tlaAngle);  // 디버깅 출력
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
            System.out.println("Tooth Number: " + toothNum);
            System.out.println("    Bounding Box Angle: " + maxBox.angle);
            System.out.println("    Bounding Box Center: " + maxBox.center);
            System.out.println("    Bounding Box Size: " + maxBox.size);
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
        // CEJ 교차점 저장을 위한 Map 초기화
        cejIntersectionsByTooth.clear();

        for (int i = 0; i < cejPoints.size(); i++) {
            List<Point> points = cejPoints.get(i);
            if (points.size() < 3) continue;

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

                Rect toothBoundingBox = Imgproc.boundingRect(toothPts);

                int minY = toothBoundingBox.y - 50;
                int maxY = toothBoundingBox.y + toothBoundingBox.height + 50;

                // 유효한 CEJ 좌표 필터링
                List<Point> validCejPoints = new ArrayList<>();
                for (Point cejPoint : points) {
                    if (toothBoundingBox.contains(cejPoint) &&
                            cejPoint.y >= minY && cejPoint.y <= maxY) {
                        validCejPoints.add(cejPoint);
                    }
                }

                if (validCejPoints.size() >= 2) {
                    // CEJ와 치아 폴리곤의 교차점 찾기
                    List<Point> intersections = findIntersectionsBetweenCEJAndTooth(validCejPoints, toothPoints);

                    // 치아 폴리곤 그리기 - 흰색
                    Imgproc.polylines(cejMappedOnlyMask, List.of(toothPts), true, new Scalar(255, 255, 255), 2);
                    Imgproc.fillPoly(cejMappedOnlyMask, List.of(toothPts), new Scalar(255, 255, 255));

                    // CEJ 라인 그리기 - 빨간색
                    MatOfPoint cejPts = new MatOfPoint();
                    cejPts.fromList(validCejPoints);
                    Imgproc.polylines(cejMappedOnlyMask, List.of(cejPts), false, new Scalar(0, 0, 255), thickness);

                    // 교차점을 cejIntersectionsByTooth Map에 저장
                    cejIntersectionsByTooth.putIfAbsent(toothNum, new ArrayList<>());
                    cejIntersectionsByTooth.get(toothNum).addAll(intersections);
                }
            }
        }

        // 교차점을 mask에 표시
        for (Map.Entry<Integer, List<Point>> entry : cejIntersectionsByTooth.entrySet()) {
            int toothNum = entry.getKey();
            List<Point> toothIntersections = entry.getValue();

            List<Point> minMaxIntersections = getMinMaxXIntersections(toothIntersections);

            System.out.println("Tooth Number: " + toothNum);


                // 교차점을 mask에 표시
                for (Point intersection : minMaxIntersections) {
                    Imgproc.circle(cejMappedOnlyMask, intersection, 5, new Scalar(0, 0, 255), -1);
                }
            }
        }


    private void drawAndMapBoneMask() {
        // Bone 교차점 저장을 위한 Map 초기화
        boneIntersectionsByTooth.clear();

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

                int minY = toothBoundingBox.y - 50;
                int maxY = toothBoundingBox.y + toothBoundingBox.height + 50;

                // 유효한 Bone 좌표 필터링
                List<Point> validBonePoints = new ArrayList<>();
                for (Point bonePoint : points) {
                    if (toothBoundingBox.contains(bonePoint) &&
                            bonePoint.y >= minY && bonePoint.y <= maxY) {
                        validBonePoints.add(bonePoint);
                    }
                }

                if (validBonePoints.size() >= 2) {
                    // Bone과 치아 폴리곤의 교차점 찾기
                    List<Point> intersections = findIntersectionsBetweenBoneAndTooth(validBonePoints, filteredToothPoints);

                    // 치아 폴리곤 그리기 - 흰색
                    Imgproc.polylines(cejMappedOnlyMask, List.of(toothPts), true, new Scalar(255, 255, 255), 2);
                    Imgproc.fillPoly(cejMappedOnlyMask, List.of(toothPts), new Scalar(255, 255, 255));

                    // Bone 라인 그리기 - 녹색
                    MatOfPoint bonePts = new MatOfPoint();
                    bonePts.fromList(validBonePoints);
                    Imgproc.polylines(cejMappedOnlyMask, List.of(bonePts), false, new Scalar(0, 255, 0), thickness);

                    // 교차점을 boneIntersectionsByTooth Map에 저장
                    boneIntersectionsByTooth.putIfAbsent(toothNum, new ArrayList<>());
                    boneIntersectionsByTooth.get(toothNum).addAll(intersections);

                    // 교차점을 mask에 표시
                    List<Point> minMaxIntersections = getMinMaxXIntersections(intersections);
                    for (Point intersection : minMaxIntersections) {
                        Imgproc.circle(cejMappedOnlyMask, intersection, 5, new Scalar(0, 255, 0), -1);
                    }
                }
            }
        }

        // 저장된 Bone 교차점 출력
        for (Map.Entry<Integer, List<Point>> entry : boneIntersectionsByTooth.entrySet()) {
            int toothNum = entry.getKey();
            List<Point> toothIntersections = entry.getValue();

            List<Point> minMaxIntersections = getMinMaxXIntersections(toothIntersections);

            System.out.println("Tooth Number: " + toothNum);
            if (!minMaxIntersections.isEmpty()) {
                System.out.println("Min X Intersection: " + minMaxIntersections.get(0));
                if (minMaxIntersections.size() > 1) {
                    System.out.println("Max X Intersection: " + minMaxIntersections.get(1));
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

    // x값 최소, 최대에 해당하는 교차점 2개만 반환
    private List<Point> getMinMaxXIntersections(List<Point> intersections) {
        if (intersections.isEmpty()) return intersections;

        Point minXPoint = intersections.get(0);
        Point maxXPoint = intersections.get(0);

        for (Point point : intersections) {
            if (point.x < minXPoint.x) {
                minXPoint = point;
            } else if (point.x > maxXPoint.x) {
                maxXPoint = point;
            }
        }

        // 최소 x값과 최대 x값 교차점 반환
        List<Point> result = new ArrayList<>();
        result.add(minXPoint);
        if (!minXPoint.equals(maxXPoint)) {
            result.add(maxXPoint);
        }
        return result;
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