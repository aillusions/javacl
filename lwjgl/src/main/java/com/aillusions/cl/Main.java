package com.aillusions.cl;

import com.aillusions.cl.demo.UsefulDevice;
import org.lwjgl.PointerBuffer;
import org.lwjgl.opencl.CL;
import org.lwjgl.opencl.CLBufferRegion;
import org.lwjgl.opencl.CLCapabilities;
import org.lwjgl.opencl.CLMemObjectDestructorCallback;
import org.lwjgl.system.Configuration;
import org.lwjgl.system.MemoryStack;

import java.io.IOException;
import java.nio.IntBuffer;
import java.util.concurrent.CountDownLatch;

import static com.aillusions.cl.demo.CLDemo.printDeviceInfo;
import static com.aillusions.cl.demo.InfoUtil.*;
import static org.lwjgl.opencl.CL10.*;
import static org.lwjgl.opencl.CL11.*;
import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.system.MemoryUtil.NULL;
import static org.lwjgl.system.MemoryUtil.memAddress;

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

    public static void main(String... args) throws IOException {

        try (MemoryStack stack = stackPush()) {
            demo(stack);
        }
    }

    private static void demo(MemoryStack stack) {

        IntBuffer errcode_ret = stack.callocInt(1);

        UsefulDevice usDev = getUsefulDevice(stack, errcode_ret);
        if (usDev == null) {
            System.out.println("No useful device found.");
            return;
        } else {
            System.out.println("Useful device: " + usDev.toString());
        }

        long context = usDev.getContext();

        int bufferSize = Integer.MAX_VALUE;

        long buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, bufferSize, errcode_ret);
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

        usDev.clear();
    }

    public static UsefulDevice getUsefulDevice(MemoryStack stack, IntBuffer errcode_ret) {
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

        System.out.println();

        for (int p = 0; p < platforms.capacity(); p++) {
            long platform = platforms.get(p);
            ctxProps.put(1, platform);

            CLCapabilities platformCaps = CL.createPlatformCapabilities(platform);
            checkCLError(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, null, pi));

            PointerBuffer devices = stack.mallocPointer(pi.get(0));
            checkCLError(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devices, (IntBuffer) null));


            for (int d = 0; d < devices.capacity(); d++) {
                long device = devices.get(d);

                CLCapabilities caps = CL.createDeviceCapabilities(device, platformCaps);

                String deviceName = getDeviceInfoStringUTF8(device, CL_DEVICE_NAME);

                if (!caps.OpenCL11
                        || deviceName.contains("CPU")
                        || deviceName.contains("Intel")
                        || deviceName.contains("Iris")) {
                    continue;
                }

                int addressBits = getDeviceInfoInt(device, CL_DEVICE_ADDRESS_BITS);

                printDeviceInfo(device, deviceName + " -->> CL_DEVICE_OPENCL_C_VERSION", CL_DEVICE_OPENCL_C_VERSION);

                return new UsefulDevice(platform, device, deviceName, ctxProps, errcode_ret, addressBits);
            }
        }

        return null;
    }
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
