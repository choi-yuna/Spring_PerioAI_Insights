package com.fas.PrtioAI_Insights.openCV;

import org.dcm4che3.data.Attributes;
import org.dcm4che3.data.Tag;
import org.dcm4che3.io.DicomInputStream;
import org.opencv.core.Point;

import java.io.File;
import java.io.IOException;

public class DicomUtil {

    // DICOM 파일에서 Pixel Spacing 값을 불러오는 메서드
    public static double[] getPixelSpacing(String dicomFilePath) throws IOException {
        File dicomFile = new File(dicomFilePath);
        try (DicomInputStream din = new DicomInputStream(dicomFile)) {
            Attributes attributes = din.readDataset(-1, -1);
            String pixelSpacingStr = attributes.getString(Tag.PixelSpacing);
            if (pixelSpacingStr != null) {
                String[] spacingParts = pixelSpacingStr.split("\\\\");
                if (spacingParts.length == 1) {
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

    // DICOM 파일에서 이미지 너비와 높이 값을 가져오는 메서드
    public static int[] getDicomDimensions(String dicomFilePath) throws IOException {
        File dicomFile = new File(dicomFilePath);
        try (DicomInputStream din = new DicomInputStream(dicomFile)) {
            Attributes attributes = din.readDataset(-1, -1);
            int width = attributes.getInt(Tag.Columns, 0);
            int height = attributes.getInt(Tag.Rows, 0);
            return new int[]{width, height};
        }
    }

    public static double calculatePhysicalDistance(Point startPixel, Point endPixel, double[] pixelSpacing, int dicomWidth, int dicomHeight, int imageWidth, int imageHeight, Point fallbackPoint) {
        if (startPixel == null || (endPixel == null && fallbackPoint == null)) {
            return Double.MAX_VALUE; // 거리 계산 불가한 경우 무한대 리턴
        }

        Point actualEndPixel = (endPixel != null) ? endPixel : fallbackPoint;

        // PictureBox와 DICOM 이미지 간의 비율 계산
        double ratioX = dicomWidth / (double) imageWidth;
        double ratioY = dicomHeight / (double) imageHeight;

        // 픽셀 거리 계산
        double pixelLengthX = Math.abs(startPixel.x - actualEndPixel.x);
        double pixelLengthY = Math.abs(startPixel.y - actualEndPixel.y);

        // 물리적 거리 계산
        double physicalLengthX = pixelLengthX * pixelSpacing[0] * ratioX;
        double physicalLengthY = pixelLengthY * pixelSpacing[1] * ratioY;

        // 총 물리적 거리 계산
        return Math.sqrt(physicalLengthX * physicalLengthX + physicalLengthY * physicalLengthY);
    }

    public static void main(String[] args) {
        try {
            String dicomFilePath = "C:/Users/fasol/OneDrive/바탕 화면/BRM 701~800/A_7_0776_01.dcm";
            double[] pixelSpacing = getPixelSpacing(dicomFilePath);
            System.out.println("Pixel Spacing: X = " + pixelSpacing[0] + ", Y = " + pixelSpacing[1]);

            // DICOM 이미지의 크기를 DICOM 파일에서 직접 가져오기
            int[] dicomDimensions = getDicomDimensions(dicomFilePath);
            int dicomWidth = dicomDimensions[0];
            int dicomHeight = dicomDimensions[1];
            System.out.println("DICOM Width: " + dicomWidth + ", Height: " + dicomHeight);

            // 화면 이미지의 크기 설정 (PictureBox 크기 등)
            int imageWidth = 3000;   // PictureBox 너비 (픽셀)
            int imageHeight = 3000;  // PictureBox 높이 (픽셀)

            // 예제 좌표 설정
            Point start = new Point(100, 150);
            Point end = new Point(200, 250);

            // 실제 물리적 거리 계산
//            double physicalDistance = calculatePhysicalDistance(start, end, pixelSpacing, dicomWidth, dicomHeight, imageWidth, imageHeight);
        } catch (IOException e) {
            System.err.println("DICOM 파일을 읽는 중 오류가 발생했습니다: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
