package com.fas.PrtioAI_Insights.service;

import org.dcm4che3.data.Attributes;
import org.dcm4che3.data.Tag;
import org.dcm4che3.io.DicomInputStream;
import org.springframework.stereotype.Service;

import java.io.File;
import java.io.IOException;

@Service
public class DicomService {

    public double[] getPixelSpacing(String dicomFilePath) throws IOException {
        // DICOM 파일을 읽어 Pixel Spacing을 추출하는 메서드
        try (DicomInputStream dicomInputStream = new DicomInputStream(new File(dicomFilePath))) {
            Attributes attributes = dicomInputStream.readDataset(-1, -1);
            String pixelSpacingString = attributes.getString(Tag.PixelSpacing);

            if (pixelSpacingString != null) {
                String[] spacingValues = pixelSpacingString.split("\\\\");
                double xSpacing = Double.parseDouble(spacingValues[0]);
                double ySpacing = Double.parseDouble(spacingValues[1]);
                return new double[]{xSpacing, ySpacing}; // x축 및 y축의 Pixel Spacing 반환
            } else {
                throw new IllegalArgumentException("Pixel Spacing 값이 DICOM 파일에 없습니다.");
            }
        }
    }
}
