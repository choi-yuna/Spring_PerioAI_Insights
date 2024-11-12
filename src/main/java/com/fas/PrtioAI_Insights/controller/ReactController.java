package com.fas.PrtioAI_Insights.controller;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;

@Controller
public class ReactController {

    @RequestMapping(value = "/{path:[^.]*}")  // 간단한 패턴
    public String forward() {
      return "forward:/index.html";
    }
}
