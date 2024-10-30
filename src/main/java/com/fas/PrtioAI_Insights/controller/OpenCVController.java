package com.fas.PrtioAI_Insights.controller;

import com.fas.PrtioAI_Insights.service.OpenCVTeethMaskService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.Map;

@RestController
@RequestMapping("/api/opencv")
public class OpenCVController {

    private final OpenCVTeethMaskService openCVTeethMaskService;

    @Autowired
    public OpenCVController(OpenCVTeethMaskService openCVTeethMaskService) {
        this.openCVTeethMaskService = openCVTeethMaskService;
    }

//    @PostMapping("/processIni")
//    public ResponseEntity<Map<String, Map<Integer, List<Point>>>> processIniFile(
//            @RequestParam("file") MultipartFile file) {
//        try {
//            Map<String, Map<Integer, List<Point>>> result = openCVTeethMaskService.processIniFile(file.getInputStream());
//            return ResponseEntity.ok(result);
//        } catch (IOException e) {
//            return ResponseEntity.status(500).body(null);
//        }
//    }
}
