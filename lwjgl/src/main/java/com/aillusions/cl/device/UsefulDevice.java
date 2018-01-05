package com.aillusions.cl.device;

import lombok.Data;
import lombok.ToString;
import org.lwjgl.PointerBuffer;
import org.lwjgl.opencl.CLContextCallback;
import org.lwjgl.system.MemoryUtil;

import java.nio.IntBuffer;

import static com.aillusions.cl.demo.InfoUtil.checkCLError;
import static org.lwjgl.opencl.CL10.clCreateContext;
import static org.lwjgl.opencl.CL10.clReleaseContext;
import static org.lwjgl.system.MemoryUtil.memUTF8;

/**
 * @author aillusions
 */
@Data
@ToString
public class UsefulDevice {
    private long platform;
    private long device;
    private String deviceName;
    private PointerBuffer ctxProps;

    private int addressBits;

    private CLContextCallback contextC;
    private CLContextCallback contextCB;
    private long context;

    public UsefulDevice(long platform, long device, String deviceName, PointerBuffer ctxProps, IntBuffer errcode_ret, int addressBits) {
        this.platform = platform;
        this.device = device;
        this.deviceName = deviceName;
        this.ctxProps = ctxProps;
        this.addressBits = addressBits;

        context = clCreateContext(ctxProps, device, contextCB = CLContextCallback.create((errinfo, private_info, cb, user_data) -> {
            System.err.println("[LWJGL] cl_context_callback");
            System.err.println("\tInfo: " + memUTF8(errinfo));
        }), MemoryUtil.NULL, errcode_ret);

        checkCLError(errcode_ret);
    }

    public void clear() {
        int errcode;

        errcode = clReleaseContext(context);
        checkCLError(errcode);

        contextCB.free();
    }
}
