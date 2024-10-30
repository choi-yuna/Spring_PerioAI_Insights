package com.fas.PrtioAI_Insights.controller;

import com.fas.PrtioAI_Insights.service.CejBoneDistancesService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.Map;

@RestController
@RequestMapping("/api/cej-bone")
public class CejBoneDistancesController {

    private final CejBoneDistancesService cejBoneDistancesService;

    @Autowired
    public CejBoneDistancesController(CejBoneDistancesService cejBoneDistancesService) {
        this.cejBoneDistancesService = cejBoneDistancesService;
    }

    /**
     * INI 파일 업로드 및 데이터 파싱 후 전체 JSON 반환
     */
    @PostMapping("/upload-ini")
    public ResponseEntity<Map<String, Object>> uploadIniFileAndGetData(@RequestParam("file") MultipartFile file) {
        try {
            String filepath = "path/to/save/" + file.getOriginalFilename();
            file.transferTo(new java.io.File(filepath));

            // INI 파일 파싱
            cejBoneDistancesService.parseIniFile(filepath);

            // 파싱된 데이터를 바탕으로 조정된 데이터 생성 및 JSON 반환
            Map<String, Object> adjustedData = cejBoneDistancesService.calculateAdjustedCejBoneDistances();
            return ResponseEntity.ok(adjustedData);

        } catch (IOException e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(Map.of("error", "INI 파일 업로드 실패: " + e.getMessage()));
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(Map.of("error", "데이터 처리 중 오류 발생: " + e.getMessage()));
        }
    }
}
