package com.aillusions.cl;

import org.lwjgl.PointerBuffer;
import org.lwjgl.opencl.CL;
import org.lwjgl.opencl.CLCapabilities;
import org.lwjgl.system.Configuration;
import org.lwjgl.system.MemoryStack;

import java.io.IOException;
import java.nio.IntBuffer;

import static com.aillusions.cl.demo.CLDemo.printDeviceInfo;
import static com.aillusions.cl.demo.InfoUtil.checkCLError;
import static org.lwjgl.opencl.CL10.*;
import static org.lwjgl.opencl.CL11.CL_DEVICE_OPENCL_C_VERSION;
import static org.lwjgl.system.MemoryStack.stackPush;

/**
 * https://www.lwjgl.org/customize
 * https://github.com/LWJGL/lwjgl3
 *
 * @author aillusions
 */
public class Main {

    static {
        Configuration.OPENCL_EXPLICIT_INIT.set(false);
    }

    private static void demo(MemoryStack stack) {
        IntBuffer pi = stack.mallocInt(1);
        checkCLError(clGetPlatformIDs(null, pi));
        if (pi.get(0) == 0) {
            throw new RuntimeException("No OpenCL platforms found.");
        }

        PointerBuffer platforms = stack.mallocPointer(pi.get(0));
        checkCLError(clGetPlatformIDs(platforms, (IntBuffer) null));

        PointerBuffer ctxProps = stack.mallocPointer(3);
        ctxProps
                .put(0, CL_CONTEXT_PLATFORM)
                .put(2, 0);

        IntBuffer errcode_ret = stack.callocInt(1);

        System.out.println();

        for (int p = 0; p < platforms.capacity(); p++) {
            long platform = platforms.get(p);
            ctxProps.put(1, platform);

            CLCapabilities platformCaps = CL.createPlatformCapabilities(platform);
            checkCLError(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, null, pi));

            PointerBuffer devices = stack.mallocPointer(pi.get(0));
            checkCLError(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devices, (IntBuffer)null));


            for (int d = 0; d < devices.capacity(); d++) {
                long device = devices.get(d);

                CLCapabilities caps = CL.createDeviceCapabilities(device, platformCaps);

                if (caps.OpenCL11) {
                    printDeviceInfo(device, "CL_DEVICE_NAME", CL_DEVICE_NAME);
                    printDeviceInfo(device, "CL_DEVICE_OPENCL_C_VERSION", CL_DEVICE_OPENCL_C_VERSION);
                    System.out.println();
                }
            }
        }
    }

    public static void main(String... args) throws IOException {

        try (MemoryStack stack = stackPush()) {
            demo(stack);
        }

/*
        try {
            try {
                CL.create();
            } catch (Exception e) {
                System.out.println("CL init nok: " + e.getMessage());
            }

            IntBuffer errcode_ret = BufferUtils.createIntBuffer(1);


            long platform = 0;

            long device = getDevice(platform, platformCaps, deviceType);
            if (device == NULL) {
                device = getDevice(platform, platformCaps, CL_DEVICE_TYPE_CPU);
            }

            PointerBuffer ctxProps = BufferUtils.createPointerBuffer(7);
            ctxProps
                    .put(CL_CONTEXT_PLATFORM)
                    .put(platform)
                    .put(NULL)
                    .flip();

            long clContext  = clCreateContext(ctxProps, device, clContextCB = CLContextCallback.create(
                    (errinfo, private_info, cb, user_data) -> log(String.format("cl_context_callback\n\tInfo: %s", memUTF8(errinfo)))
            ), NULL, errcode_ret);
            checkCLError(errcode_ret);



            PointerBuffer strings = BufferUtils.createPointerBuffer(1);
            PointerBuffer lengths = BufferUtils.createPointerBuffer(1);

            ByteBuffer  source = ioResourceToByteBuffer("OpenCLSum.cl", 4096);
            strings.put(0, source);
            lengths.put(0, source.remaining());

            long clProgram = clCreateProgramWithSource(clContext, strings, lengths, errcode_ret);
            checkCLError(errcode_ret);

        } finally {
            CL.destroy();
        }*/
    }
}
