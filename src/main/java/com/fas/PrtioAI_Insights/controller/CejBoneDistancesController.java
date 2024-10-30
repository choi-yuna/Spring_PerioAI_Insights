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
import java.util.UUID;

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

            // UUID를 사용하여 고유한 파일 이름 생성
            String uniqueFileName = UUID.randomUUID().toString() + "_" + file.getName();
            Path filePath = Paths.get(tempDir, uniqueFileName);
            File tempFile = filePath.toFile();

            // 파일 저장
            System.out.println("Saving file to: " + tempFile.getAbsolutePath());
            file.transferTo(tempFile);
            System.out.println("File saved successfully.");

            // 저장된 파일 경로에서 INI 파일을 파싱
            System.out.println("Parsing INI file: " + filePath.toString());
            cejBoneDistancesService.parseIniFile(filePath.toString());
            System.out.println("INI file parsed successfully.");

            // 조정된 데이터 생성 및 JSON 반환
            Map<Integer, Map<String, Object>> adjustedData = cejBoneDistancesService.calculateAdjustedCejBoneDistances();
            return ResponseEntity.ok(adjustedData);

        } catch (IOException e) {
            System.err.println("INI 파일 업로드 실패: " + e.getMessage());
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(Map.of("error", "INI 파일 업로드 실패: " + e.getMessage()));
        } catch (Exception e) {
            System.err.println("데이터 처리 중 오류 발생: " + e.getMessage());
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(Map.of("error", "데이터 처리 중 오류 발생: " + e.getMessage()));
        }
    }

}
