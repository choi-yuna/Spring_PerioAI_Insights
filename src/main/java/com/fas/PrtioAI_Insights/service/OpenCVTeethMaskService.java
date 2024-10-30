package com.fas.PrtioAI_Insights.service;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.springframework.stereotype.Service;

import java.io.*;
import java.util.*;

@Service
public class OpenCVTeethMaskService {
    // 데이터 초기화 코드 (기존 OpenCVTeethMask 클래스와 동일)
    private static List<Integer> teethNum = new ArrayList<>();
    private static List<List<Point>> teethPoints = new ArrayList<>();
    private static List<Integer> teethSize = new ArrayList<>();
    private static List<List<Point>> cejPoints = new ArrayList<>();
    private static List<Integer> cejSize = new ArrayList<>();
    private static List<List<Point>> bonePoints = new ArrayList<>();
    private static List<Integer> boneSize = new ArrayList<>();
    private static Map<Integer, List<Point>> teethCejPoints = new HashMap<>();
    private static Map<Integer, List<Point>> bonePointsByNum = new HashMap<>();

    // ini 파일을 파싱하고 데이터를 처리하는 메서드
    public Map<String, Map<Integer, List<Point>>> processIniFile(InputStream iniInputStream) throws IOException {
        parseIniFile(iniInputStream);
        drawAndMapCejMask();
        drawAndMapBoneMask();

        // JSON 형태의 데이터를 반환
        Map<String, Map<Integer, List<Point>>> result = new HashMap<>();
        result.put("cej", teethCejPoints);
        result.put("bone", bonePointsByNum);

        return result;
    }

    private void parseIniFile(InputStream inputStream) throws IOException {
        BufferedReader br = new BufferedReader(new InputStreamReader(inputStream));
        String line;
        List<Point> loadedPoints = new ArrayList<>();
        int _Size = 0;
        String type_ = "";
        int num = 0;

        while ((line = br.readLine()) != null) {
            line = line.trim();
            if (line.startsWith("N=")) {
                num = Integer.parseInt(line.substring(2));
            } else if (line.startsWith("END")) {
                if ("C".equals(type_)) {
                    cejPoints.add(new ArrayList<>(loadedPoints));
                    cejSize.add(_Size);
                } else if ("D".equals(type_)) {
                    bonePoints.add(new ArrayList<>(loadedPoints));
                    boneSize.add(_Size);
                }
                loadedPoints.clear();
            } else if (line.startsWith("P=")) {
                String[] parts = line.substring(2).split(",");
                int x = Integer.parseInt(parts[0].trim());
                int y = Integer.parseInt(parts[1].trim());
                loadedPoints.add(new Point(x, y));
            } else if (line.startsWith("TD")) {
                type_ = "C";
            } else if (line.startsWith("BD")) {
                type_ = "D";
            }
        }
        br.close();
    }

    private void drawAndMapCejMask() {
        for (List<Point> points : cejPoints) {
            for (int j = 0; j < teethPoints.size(); j++) {
                MatOfPoint toothPts = new MatOfPoint();
                toothPts.fromList(teethPoints.get(j));
                for (Point cejPoint : points) {
                    if (Imgproc.pointPolygonTest(new MatOfPoint2f(toothPts.toArray()), cejPoint, true) >= -10.0) {
                        int toothNum = teethNum.get(j);
                        teethCejPoints.computeIfAbsent(toothNum, k -> new ArrayList<>()).add(cejPoint);
                    }
                }
            }
        }
    }

    private void drawAndMapBoneMask() {
        for (List<Point> points : bonePoints) {
            for (int j = 0; j < teethPoints.size(); j++) {
                MatOfPoint toothPts = new MatOfPoint();
                toothPts.fromList(teethPoints.get(j));
                for (Point bonePoint : points) {
                    if (Imgproc.pointPolygonTest(new MatOfPoint2f(toothPts.toArray()), bonePoint, true) >= -10.0) {
                        int toothNum = teethNum.get(j);
                        bonePointsByNum.computeIfAbsent(toothNum, k -> new ArrayList<>()).add(bonePoint);
                    }
                }
            }
        }
    }
}
