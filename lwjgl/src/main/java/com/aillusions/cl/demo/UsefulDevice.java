package com.aillusions.cl.demo;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.lwjgl.PointerBuffer;

/**
 * @author aillusions
 */
@Data
@AllArgsConstructor
public class UsefulDevice {
    private long platform;
    private long device;
    private String deviceName;
    private PointerBuffer ctxProps;
}
