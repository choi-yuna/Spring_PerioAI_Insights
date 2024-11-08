package com.fas.PrtioAI_Insights.openCV;

import org.dcm4che3.data.Attributes;
import org.dcm4che3.data.Tag;
import org.dcm4che3.io.DicomInputStream;
import org.opencv.core.Point;

import java.io.File;
import java.io.IOException;

public class Dicom {

    // DICOM 파일에서 Pixel Spacing 값을 불러오는 메서드
    public static double[] getPixelSpacing(String dicomFilePath) throws IOException {
        File dicomFile = new File(dicomFilePath);
        try (DicomInputStream din = new DicomInputStream(dicomFile)) {
            Attributes attributes = din.readDataset(-1, -1);
            String pixelSpacingStr = attributes.getString(Tag.PixelSpacing);
            if (pixelSpacingStr != null) {
                String[] spacingParts = pixelSpacingStr.split("\\\\");
                if (spacingParts.length == 1) {
                    // 단일 값이 제공된 경우, X와 Y 모두 동일한 값으로 사용
                    double pixelSpacing = Double.parseDouble(spacingParts[0]);
                    return new double[]{pixelSpacing, pixelSpacing};
                } else if (spacingParts.length >= 2) {
                    double pixelSpacingX = Double.parseDouble(spacingParts[0]);
                    double pixelSpacingY = Double.parseDouble(spacingParts[1]);
                    return new double[]{pixelSpacingX, pixelSpacingY};
                } else {
                    throw new IllegalArgumentException("DICOM 파일에 Pixel Spacing 값이 잘못되었습니다: " + pixelSpacingStr);
                }
            } else {
                throw new IllegalArgumentException("DICOM 파일에 Pixel Spacing 정보가 없습니다.");
            }
        }
    }



    // 픽셀 간 거리를 계산하는 메서드
    public static double calculatePixelDistance(Point start, Point end, double[] pixelSpacing) {
        double pixelDistanceX = Math.abs(end.x - start.x);
        double pixelDistanceY = Math.abs(end.y - start.y);

        // 실제 물리적 거리로 변환
        double physicalDistanceX = pixelDistanceX * pixelSpacing[0];
        double physicalDistanceY = pixelDistanceY * pixelSpacing[1];

        // 총 거리 계산 (피타고라스 정리)
        return Math.sqrt(physicalDistanceX * physicalDistanceX + physicalDistanceY * physicalDistanceY);
    }

    public static void main(String[] args) {
        try {
            // DICOM 파일 경로 설정
            String dicomFilePath = "C:/Users/fasol/OneDrive/바탕 화면/BRM 701~800/A_7_0776_01.dcm" ;
            double[] pixelSpacing = getPixelSpacing(dicomFilePath);
            System.out.println("Pixel Spacing: X = " + pixelSpacing[0] + ", Y = " + pixelSpacing[1]);

            // 예제 좌표 설정
            Point start = new Point(100, 150);
            Point end = new Point(200, 250);

            // 픽셀 간 물리적 거리 계산
            double physicalDistance = calculatePixelDistance(start, end, pixelSpacing);
            System.out.println("픽셀 간 물리적 거리: " + physicalDistance + " mm");
        } catch (IOException e) {
            System.err.println("DICOM 파일을 읽는 중 오류가 발생했습니다: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
