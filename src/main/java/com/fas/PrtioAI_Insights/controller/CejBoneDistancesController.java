package com.fas.PrtioAI_Insights.controller;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fas.PrtioAI_Insights.service.CejBoneDistancesService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;
import java.util.UUID;

@RestController
@RequestMapping("/api")
public class CejBoneDistancesController {

    private final CejBoneDistancesService cejBoneDistancesService;
    private final ObjectMapper objectMapper;
    private String jsonFileContent;

    @Autowired
    public CejBoneDistancesController(CejBoneDistancesService cejBoneDistancesService, ObjectMapper objectMapper) {
        this.cejBoneDistancesService = cejBoneDistancesService;
        this.objectMapper = objectMapper;
    }

    /**
     * INI 파일과 JSON 파일을 임시 디렉토리에 저장하고 파싱 후 JSON 데이터 반환
     */
    @PostMapping("/upload-ini-json")
    public ResponseEntity<?> uploadIniAndJsonFiles(@RequestParam("iniFile") MultipartFile iniFile,
                                                   @RequestParam("jsonFile") MultipartFile jsonFile) {
        try {
            // 임시 디렉토리에 INI 파일과 JSON 파일 저장
            String tempDir = System.getProperty("java.io.tmpdir");

            // 고유 파일 이름 생성 (UUID + 원래 확장자)
            String iniFileName = UUID.randomUUID() + "_ini.ini";
            String jsonFileName = UUID.randomUUID() + "_json.json";

            Path iniFilePath = Paths.get(tempDir, iniFileName);
            Path jsonFilePath = Paths.get(tempDir, jsonFileName);

            // 파일을 저장할 디렉토리가 있는지 확인하고 없으면 생성
            if (!Files.exists(iniFilePath.getParent())) {
                Files.createDirectories(iniFilePath.getParent());
            }
            if (!Files.exists(jsonFilePath.getParent())) { // JSON 파일 경로 확인 및 생성
                Files.createDirectories(jsonFilePath.getParent());
            }

            // INI 파일 저장
            File iniTempFile = iniFilePath.toFile();
            iniFile.transferTo(iniTempFile);
            System.out.println("INI file saved successfully to: " + iniTempFile.getAbsolutePath());

            // JSON 파일 저장
            File jsonTempFile = jsonFilePath.toFile();
            jsonFile.transferTo(jsonTempFile);
            System.out.println("JSON file saved successfully to: " + jsonTempFile.getAbsolutePath());

            // INI 파일을 파싱
            System.out.println("Parsing INI file: " + iniFilePath.toString());
            cejBoneDistancesService.parseIniFile(iniFilePath.toString());
            System.out.println("INI file parsed successfully.");

            // JSON 파일 내용을 읽어와 문자열로 저장
            System.out.println("Reading JSON file content: " + jsonFilePath.toString());
            jsonFileContent = new String(Files.readAllBytes(jsonFilePath));
            System.out.println("JSON file content saved successfully.");

            // 조정된 데이터 생성 및 JSON 반환
            Map<Integer, Map<String, Object>> adjustedData = cejBoneDistancesService.calculateAdjustedCejBoneDistances();
            return ResponseEntity.ok(adjustedData);

        } catch (IOException e) {
            System.err.println("파일 업로드 실패: " + e.getMessage());
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(Map.of("error", "파일 업로드 실패: " + e.getMessage()));
        } catch (Exception e) {
            System.err.println("데이터 처리 중 오류 발생: " + e.getMessage());
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(Map.of("error", "데이터 처리 중 오류 발생: " + e.getMessage()));
        }
    }
}
