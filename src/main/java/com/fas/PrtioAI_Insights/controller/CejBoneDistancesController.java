package com.fas.PrtioAI_Insights.controller;

import com.fas.PrtioAI_Insights.service.CejBoneDistancesService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;

@RestController
@RequestMapping("/api")
public class CejBoneDistancesController {

    private final CejBoneDistancesService cejBoneDistancesService;

    @Autowired
    public CejBoneDistancesController(CejBoneDistancesService cejBoneDistancesService) {
        this.cejBoneDistancesService = cejBoneDistancesService;
    }

    /**
     * INI 파일을 임시 디렉토리에 저장하고 파싱 후 JSON 데이터 반환
     */
    @PostMapping("/upload-ini")
    public ResponseEntity<?> uploadIniFileAndGetData(@RequestParam("file") MultipartFile file) {
        try {
            // 시스템 임시 디렉토리에 저장할 파일 경로 설정
            String tempDir = System.getProperty("java.io.tmpdir");
            Path filePath = Paths.get(tempDir, file.getOriginalFilename());
            File tempFile = filePath.toFile();

            // 파일 저장
            file.transferTo(tempFile);

            // 저장된 파일 경로에서 INI 파일을 파싱
            cejBoneDistancesService.parseIniFile(filePath.toString());

            // 조정된 데이터 생성 및 JSON 반환
            Map<Integer, Map<String, Object>> adjustedData = cejBoneDistancesService.calculateAdjustedCejBoneDistances();
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
