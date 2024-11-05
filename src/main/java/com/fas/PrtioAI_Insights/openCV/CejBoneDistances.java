package com.fas.PrtioAI_Insights.openCV;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Moments;

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
    private static Map<Integer, RotatedRect> maxBoundingBoxMap = new HashMap<>();

    // 각 치아 번호별 모든 좌표를 합친 결과를 저장하기 위한 맵
    private static Map<Integer, List<Point>> allPointsByTooth = new HashMap<>();

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
            drawTlaMask();
            drawCombinedMask();

            Map<Integer, List<Point>> intersectionsByTooth = findAndMarkIntersections();
            printIntersectionsByTooth(intersectionsByTooth);

            removeIslands(bimasks, 900);

            Imgcodecs.imwrite("Combined_Teeth_Mask.png", combinedMask);
            Imgcodecs.imwrite("cejMask.png", cejMask);
            Imgcodecs.imwrite("mappedCejMask.png", mappedCejMask);
            Imgcodecs.imwrite("tlaMask.png", tlaMask);
            Imgcodecs.imwrite("boneMask.png", boneMask);
            Imgcodecs.imwrite("cejMappedOnly.png", cejMappedOnlyMask);
            Imgcodecs.imwrite("boneMappedOnly.png", boneMappedOnlyMask);
            Imgcodecs.imwrite("Combined_Mask.png", combinedMask);

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

    private static Map<Integer, List<Point>> findAndMarkIntersections() {
        Map<Integer, List<Point>> intersectionsByTooth = new HashMap<>();

        // TLA와 CEJ, TLA와 Bone, CEJ와 Bone 교차점을 탐색
        for (Map.Entry<Integer, List<List<Point>>> entry : tlaPointsByNum.entrySet()) {
            int toothNum = entry.getKey();
            List<List<Point>> tlaSegments = entry.getValue();

            for (List<Point> tlaSegment : tlaSegments) {
                if (tlaSegment.size() >= 2) {
                    List<Point> cejIntersections = findIntersections(Map.of(toothNum, tlaSegment), teethCejPoints, new Scalar(255, 0, 0));
                    List<Point> boneIntersections = findIntersections(Map.of(toothNum, tlaSegment), bonePointsByNum, new Scalar(0, 255, 255));

                    intersectionsByTooth.computeIfAbsent(toothNum, k -> new ArrayList<>()).addAll(cejIntersections);
                    intersectionsByTooth.computeIfAbsent(toothNum, k -> new ArrayList<>()).addAll(boneIntersections);
                }
            }
        }

        List<Point> cejBoneIntersections = findIntersections(teethCejPoints, bonePointsByNum, new Scalar(0, 0, 255)); // CEJ와 Bone 교차점
        for (Map.Entry<Integer, List<Point>> entry : teethCejPoints.entrySet()) {
            int toothNum = entry.getKey();
            intersectionsByTooth.computeIfAbsent(toothNum, k -> new ArrayList<>()).addAll(cejBoneIntersections);
        }

        return intersectionsByTooth;
    }

    private static List<Point> findIntersections(
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

            System.out.println("치아 번호: " + toothNum + " - 교차점 좌표:");
            for (Point intersection : intersections) {
                System.out.println("    " + intersection);
            }
        }
    }
    private static void calculateAndPrintAllDistancesToBoundingBox(RotatedRect boundingBox, List<Point> cejPoints, int toothNum) {
        // 바운딩 박스의 네 꼭짓점 계산
        Point[] boxPoints = new Point[4];
        boundingBox.points(boxPoints);

        // 상단과 하단 라인 정의 (RotatedRect를 기준으로)
        Point topLeft = boxPoints[0];
        Point topRight = boxPoints[1];
        Point bottomRight = boxPoints[2];
        Point bottomLeft = boxPoints[3];

        System.out.println("치아 번호: " + toothNum + " - 바운딩 박스와 CEJ 사이의 모든 거리:");

        // 상악과 하악을 구분하기 위해 치아 번호를 사용
        boolean isMaxillary = (toothNum >= 11 && toothNum <= 28); // 상악 치아 번호
        boolean isMandibular = (toothNum >= 31 && toothNum <= 48); // 하악 치아 번호

        // 상단과 하단 선의 기울기와 y절편 계산
        double topSlope = (topRight.y - topLeft.y) / (topRight.x - topLeft.x);
        double topIntercept = topLeft.y - topSlope * topLeft.x;

        double bottomSlope = (bottomRight.y - bottomLeft.y) / (bottomRight.x - bottomLeft.x);
        double bottomIntercept = bottomLeft.y - bottomSlope * bottomLeft.x;

        for (Point cejPoint : cejPoints) {
            // CEJ 점의 x 좌표에 따라 상단 또는 하단 y 좌표를 계산

            // 선형 방정식을 사용하여 주어진 x 좌표에서 상단과 하단 y 값 계산
            double topY = topSlope * cejPoint.x + topIntercept;
            double bottomY = bottomSlope * cejPoint.x + bottomIntercept;

            // 바운딩 박스 x 범위 내의 CEJ 점과 바운딩 박스 경계 사이의 거리 계산
            if (cejPoint.x >= Math.min(topLeft.x, topRight.x) && cejPoint.x <= Math.max(topLeft.x, topRight.x)) {
                double distance;
                if (isMaxillary) {
                    // 상악일 경우 bottomY를 기준으로 거리 계산
                    distance = Math.abs(cejPoint.y - bottomY);
                    System.out.println("    상악 치아 - CEJ 좌표: " + cejPoint + " -> 바운딩 박스 하단과의 거리: " + distance);
                } else if (isMandibular) {
                    // 하악일 경우 topY를 기준으로 거리 계산
                    distance = Math.abs(cejPoint.y - topY);
                    System.out.println("    하악 치아 - CEJ 좌표: " + cejPoint + " -> 바운딩 박스 상단과의 거리: " + distance);
                }
            }
        }
    }

    // 두 점 사이의 선형 보간을 사용하여 특정 x에서의 y 좌표를 계산
    private static double interpolateY(Point p1, Point p2, double x) {
        if (p1.x == p2.x) {
            return p1.y;  // 수직선의 경우 p1.y 값을 반환
        }
        // 선형 보간을 통해 y 좌표 계산
        return p1.y + (p2.y - p1.y) * (x - p1.x) / (p2.x - p1.x);
    }



// CEJ, Bone, TLA를 그리기 위한 기존 코드에 바운딩 박스를 추가하여 전체 코드를 수정합니다.

    private static void drawCombinedMask() {
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
            List<Point> cejPointsForTooth = teethCejPoints.get(toothNum);
            if (cejPointsForTooth != null) {
                calculateAndPrintAllDistancesToBoundingBox(maxBox, cejPointsForTooth, toothNum);
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

    private static void drawTeethMasks() {


        // 각 치아 번호에 대해 좌표를 수집
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

            // 면적 필터링
            if (area < 900) {
                continue;
            }

            // 각 치아 번호에 해당하는 좌표를 합침
            allPointsByTooth.computeIfAbsent(toothNum, k -> new ArrayList<>()).addAll(points);

            // 폴리곤 그리기
            Imgproc.polylines(combinedMask, List.of(pts), true, new Scalar(255, 255, 255), thickness);
            Imgproc.fillPoly(combinedMask, List.of(pts), new Scalar(255, 255, 255));
        }

        // 각 치아 번호별 최대 및 최소 Y 좌표 계산
        for (Map.Entry<Integer, List<Point>> entry : allPointsByTooth.entrySet()) {
            int toothNum = entry.getKey();
            List<Point> combinedPoints = entry.getValue();

            double minY = Double.MAX_VALUE;
            double maxY = Double.MIN_VALUE;

            for (Point p : combinedPoints) {
                if (p.y < minY) minY = p.y;
                if (p.y > maxY) maxY = p.y;
            }

            // 상악은 maxY, 하악은 minY를 기준 Y 좌표로 저장
            if (toothNum >= 11 && toothNum <= 28) {
                yReferenceByTooth.put(toothNum, maxY); // 상악 최대 Y
            } else if (toothNum >= 31 && toothNum <= 48) {
                yReferenceByTooth.put(toothNum, minY); // 하악 최소 Y
            }

            System.out.println("치아 번호: " + toothNum + " - 최대 Y: " + maxY + ", 최소 Y: " + minY);
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

        private static void drawAndMapBoneMask() {
        for (int i = 0; i < bonePoints.size(); i++) {
            List<Point> points = bonePoints.get(i);
            if (points.size() < 3) {
                continue;}

            MatOfPoint pts = new MatOfPoint();
            pts.fromList(points);
            int thickness = boneSize.get(i);

            double area = Imgproc.contourArea(pts);
            if (area < 300 || thickness > 2) {
                continue;}

            for (Map.Entry<Integer, List<Point>> entry : allPointsByTooth.entrySet()) {
                int toothNum = entry.getKey();
                List<Point> filteredToothPoints = entry.getValue();

                if (filteredToothPoints == null || filteredToothPoints.size() < 3) {
                    continue;
                }

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

    private static void drawTlaMask() {
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

                    // 필터링된 TLA 좌표 출력
                    System.out.println("치아 번호: " + toothNum + " - 필터링된 TLA 좌표: " + filteredTlaPoints);
                }
            }
        }
    }
}

