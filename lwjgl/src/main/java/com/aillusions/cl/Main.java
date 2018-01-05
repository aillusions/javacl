package com.aillusions.cl;

import org.lwjgl.PointerBuffer;
import org.lwjgl.opencl.*;
import org.lwjgl.system.Configuration;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;

import java.io.IOException;
import java.nio.IntBuffer;
import java.util.concurrent.CountDownLatch;

import static com.aillusions.cl.demo.CLDemo.printDeviceInfo;
import static com.aillusions.cl.demo.InfoUtil.checkCLError;
import static com.aillusions.cl.demo.InfoUtil.getDeviceInfoStringUTF8;
import static org.lwjgl.opencl.CL10.*;
import static org.lwjgl.opencl.CL11.*;
import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.system.MemoryUtil.*;

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
            checkCLError(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devices, (IntBuffer) null));

            int numDevicesTested = 0;
            for (int d = 0; d < devices.capacity(); d++) {
                long device = devices.get(d);

                CLCapabilities caps = CL.createDeviceCapabilities(device, platformCaps);

                String deviceName = getDeviceInfoStringUTF8(device, CL_DEVICE_NAME);

                if (numDevicesTested > 0
                        || !caps.OpenCL11
                        || deviceName.contains("CPU")
                        || deviceName.contains("Iris")) {
                    continue;
                }

                numDevicesTested++;

                printDeviceInfo(device, deviceName + " -->> CL_DEVICE_OPENCL_C_VERSION", CL_DEVICE_OPENCL_C_VERSION);

                CLContextCallback contextCB;
                long context = clCreateContext(ctxProps, device, contextCB = CLContextCallback.create((errinfo, private_info, cb, user_data) -> {
                    System.err.println("[LWJGL] cl_context_callback");
                    System.err.println("\tInfo: " + memUTF8(errinfo));
                }), MemoryUtil.NULL, errcode_ret);
                checkCLError(errcode_ret);

                long buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, 128000000, errcode_ret);
                checkCLError(errcode_ret);

                CLMemObjectDestructorCallback bufferCB1;
                CLMemObjectDestructorCallback bufferCB2;

                long subbuffer;

                CLMemObjectDestructorCallback subbufferCB;

                int errcode;

                CountDownLatch destructorLatch;

                destructorLatch = new CountDownLatch(3);

                errcode = clSetMemObjectDestructorCallback(buffer, bufferCB1 = CLMemObjectDestructorCallback.create((memobj, user_data) -> {
                    System.out.println("\t\tBuffer destructed (1): " + memobj);
                    destructorLatch.countDown();
                }), NULL);
                checkCLError(errcode);

                errcode = clSetMemObjectDestructorCallback(buffer, bufferCB2 = CLMemObjectDestructorCallback.create((memobj, user_data) -> {
                    System.out.println("\t\tBuffer destructed (2): " + memobj);
                    destructorLatch.countDown();
                }), NULL);
                checkCLError(errcode);

                try (CLBufferRegion buffer_region = CLBufferRegion.malloc()) {
                    buffer_region.origin(0);
                    buffer_region.size(64);

                    subbuffer = nclCreateSubBuffer(buffer,
                            CL_MEM_READ_ONLY,
                            CL_BUFFER_CREATE_TYPE_REGION,
                            buffer_region.address(),
                            memAddress(errcode_ret));
                    checkCLError(errcode_ret);
                }

                errcode = clSetMemObjectDestructorCallback(subbuffer, subbufferCB = CLMemObjectDestructorCallback.create((memobj, user_data) -> {
                    System.out.println("\t\tSub Buffer destructed: " + memobj);
                    destructorLatch.countDown();
                }), NULL);
                checkCLError(errcode);


                //
                //
                //


                System.out.println();

                if (subbuffer != NULL) {
                    errcode = clReleaseMemObject(subbuffer);
                    checkCLError(errcode);
                }

                errcode = clReleaseMemObject(buffer);
                checkCLError(errcode);

                // mem object destructor callbacks are called asynchronously on Nvidia

                try {
                    destructorLatch.await();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }

                subbufferCB.free();

                bufferCB2.free();
                bufferCB1.free();

                errcode = clReleaseContext(context);
                checkCLError(errcode);

                contextCB.free();
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
