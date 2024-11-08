package com.fas.PrtioAI_Insights.controller;

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
import java.util.List;
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
     * INI 파일과 JSON 파일을 임시 디렉토리에 저장하고 파싱 후 JSON 데이터 반환
     */
    @PostMapping("/upload-ini-json")
    public ResponseEntity<?> uploadIniAndJsonFiles(@RequestParam("iniFile") MultipartFile iniFile,
                                                   @RequestParam("jsonFile") MultipartFile jsonFile,
                                                   @RequestParam("dcmFile") MultipartFile dcmFile) {
        try {
            // 임시 디렉토리에 INI 파일, JSON 파일, DICOM 파일 저장
            String tempDir = System.getProperty("java.io.tmpdir");

            // 고유 파일 이름 생성 (UUID + 원래 확장자)
            String iniFileName = UUID.randomUUID() + "_ini.ini";
            String jsonFileName = UUID.randomUUID() + "_json.json";
            String dcmFileName = UUID.randomUUID() + "_dcm.dcm";

            Path iniFilePath = Paths.get(tempDir, iniFileName);
            Path jsonFilePath = Paths.get(tempDir, jsonFileName);
            Path dcmFilePath = Paths.get(tempDir, dcmFileName);

            // 파일을 저장할 디렉토리가 있는지 확인하고 없으면 생성
            Files.createDirectories(iniFilePath.getParent());
            Files.createDirectories(jsonFilePath.getParent());
            Files.createDirectories(dcmFilePath.getParent());

            // INI 파일 저장
            File iniTempFile = iniFilePath.toFile();
            iniFile.transferTo(iniTempFile);
            System.out.println("INI file saved successfully to: " + iniTempFile.getAbsolutePath());

            // JSON 파일 저장
            File jsonTempFile = jsonFilePath.toFile();
            jsonFile.transferTo(jsonTempFile);
            System.out.println("JSON file saved successfully to: " + jsonTempFile.getAbsolutePath());

            // DICOM 파일 저장
            File dcmTempFile = dcmFilePath.toFile();
            dcmFile.transferTo(dcmTempFile);
            System.out.println("DICOM file saved successfully to: " + dcmTempFile.getAbsolutePath());

            // INI 파일 파싱
            System.out.println("Parsing INI file: " + iniFilePath);
            cejBoneDistancesService.parseIniFile(iniFilePath.toString());
            System.out.println("INI file parsed successfully.");

            Map<Integer, Map<String, List<Double>>> adjustedData = Map.of();
            try {
                // JSON 파일 경로를 통해 정상 치아 번호를 필터링하고 조정된 데이터를 생성
                System.out.println("Calculating adjusted distances based on JSON file content...");
                adjustedData = cejBoneDistancesService.calculateDistances(jsonFilePath.toString(), dcmFilePath.toString());
                System.out.println("Adjusted data generated successfully.");
            } catch (IOException e) {
                System.err.println("IOException 발생: " + e.getMessage());
                e.printStackTrace();
            } catch (NullPointerException e) {
                System.err.println("NullPointerException 발생: " + e.getMessage());
                e.printStackTrace();
            } catch (Exception e) {
                System.err.println("예기치 않은 오류 발생: " + e.getMessage());
                e.printStackTrace();
            }

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

