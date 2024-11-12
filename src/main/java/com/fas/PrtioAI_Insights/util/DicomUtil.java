package com.fas.PrtioAI_Insights.openCV;

import org.dcm4che3.data.Attributes;
import org.dcm4che3.data.Tag;
import org.dcm4che3.io.DicomInputStream;
import org.opencv.core.Point;

import java.io.File;
import java.io.IOException;

/**
 * DICOM 파일을 다루기 위한 유틸리티 클래스입니다. 이 클래스는 DICOM 파일에서 픽셀 간격, 이미지 크기 등의 정보를 가져오며,
 * 주어진 좌표 간의 실제 물리적 거리를 계산하는 기능을 제공합니다.
 */
public class DicomUtil {

    /**
     * DICOM 파일에서 픽셀 간격 정보를 가져오는 메서드입니다.
     *
     * @param dicomFilePath DICOM 파일의 경로
     * @return [X 방향 픽셀 간격, Y 방향 픽셀 간격]
     * @throws IOException DICOM 파일을 읽는 중 오류가 발생한 경우
     */
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

    /**
     * DICOM 파일에서 이미지의 너비와 높이 정보를 가져오는 메서드입니다.
     *
     * @param dicomFilePath DICOM 파일의 경로
     * @return [너비, 높이]
     * @throws IOException DICOM 파일을 읽는 중 오류가 발생한 경우
     */
    public static int[] getDicomDimensions(String dicomFilePath) throws IOException {
        File dicomFile = new File(dicomFilePath);
        try (DicomInputStream din = new DicomInputStream(dicomFile)) {
            Attributes attributes = din.readDataset(-1, -1);
            int width = attributes.getInt(Tag.Columns, 0);
            int height = attributes.getInt(Tag.Rows, 0);
            return new int[]{width, height};
        }
    }

    /**
     * 주어진 좌표들 간의 물리적 거리를 계산하는 메서드입니다.
     *
     * @param startPixel 시작 픽셀 좌표
     * @param endPixel 종료 픽셀 좌표 (없을 경우 fallbackPoint 사용)
     * @param pixelSpacing [X 방향 픽셀 간격, Y 방향 픽셀 간격]
     * @param dicomWidth DICOM 이미지의 너비
     * @param dicomHeight DICOM 이미지의 높이
     * @param imageWidth 화면 이미지의 너비
     * @param imageHeight 화면 이미지의 높이
     * @param fallbackPoint 종료 픽셀 좌표가 없을 때 대신 사용할 좌표
     * @return 실제 물리적 거리
     */
    public static double calculatePhysicalDistance(Point startPixel, Point endPixel, double[] pixelSpacing,
                                                   int dicomWidth, int dicomHeight, int imageWidth, int imageHeight,
                                                   Point fallbackPoint) {
        if (startPixel == null || (endPixel == null && fallbackPoint == null)) {
            return Double.MAX_VALUE; // 거리 계산 불가한 경우 무한대 리턴
        }

        Point actualEndPixel = (endPixel != null) ? endPixel : fallbackPoint;

        // DICOM 이미지와 화면 이미지 간의 비율 계산
        double ratioX = dicomWidth / (double) imageWidth;
        double ratioY = dicomHeight / (double) imageHeight;

        // 픽셀 간 거리 계산
        double pixelLengthX = Math.abs(startPixel.x - actualEndPixel.x);
        double pixelLengthY = Math.abs(startPixel.y - actualEndPixel.y);

        // 물리적 거리 계산
        double physicalLengthX = pixelLengthX * pixelSpacing[0] * ratioX;
        double physicalLengthY = pixelLengthY * pixelSpacing[1] * ratioY;

        // 최종 물리적 거리 계산
        return Math.sqrt(physicalLengthX * physicalLengthX + physicalLengthY * physicalLengthY);
    }
 }
