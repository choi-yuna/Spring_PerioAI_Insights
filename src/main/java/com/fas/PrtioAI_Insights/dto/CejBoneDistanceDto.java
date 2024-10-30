package com.fas.PrtioAI_Insights.dto;

import lombok.Data;
import lombok.Getter;
import org.opencv.core.Point;

import java.util.ArrayList;
import java.util.List;

@Data
public class CejBoneDistanceDto {
    private int toothNumber;
    private List<Point> cejPoints = new ArrayList<>();
    private List<Point> bonePoints = new ArrayList<>();
    private List<Point> adjustedCejPoints = new ArrayList<>();
    private List<Point> adjustedBonePoints = new ArrayList<>();
    private List<Double> cejDistances = new ArrayList<>();
    private List<Double> boneDistances = new ArrayList<>();

    public void calculateAdjustedCoordinatesAndDistances(double minY) {
        for (Point cej : cejPoints) {
            double adjustedY = cej.y - minY;
            Point adjustedCej = new Point(cej.x, adjustedY);
            adjustedCejPoints.add(adjustedCej);
            cejDistances.add(Math.abs(adjustedY));
        }

        for (Point bone : bonePoints) {
            double adjustedY = bone.y - minY;
            Point adjustedBone = new Point(bone.x, adjustedY);
            adjustedBonePoints.add(adjustedBone);
            boneDistances.add(Math.abs(adjustedY));
        }
    }

    // Getters and Setters for all fields
}
