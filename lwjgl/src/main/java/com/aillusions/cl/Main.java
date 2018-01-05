package com.aillusions.cl;

import com.aillusions.cl.device.UsefulDevice;
import com.aillusions.cl.device.OpenCLDeviceProvider;
import org.lwjgl.BufferUtils;
import org.lwjgl.PointerBuffer;
import org.lwjgl.opencl.*;
import org.lwjgl.system.Configuration;
import org.lwjgl.system.MemoryStack;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

import static com.aillusions.cl.demo.CLDemo.getEventStatusName;
import static com.aillusions.cl.demo.InfoUtil.*;
import static org.lwjgl.opencl.CL10.*;
import static org.lwjgl.opencl.CL11.*;
import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.system.MemoryUtil.*;
import static org.lwjgl.system.Pointer.POINTER_SIZE;

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

        UsefulDevice usDev = OpenCLDeviceProvider.getUsefulDevice(stack, errcode_ret);
        if (usDev == null) {
            System.out.println("No useful device found.");
            return;
        } else {
            System.out.println("Useful device: " + usDev.toString());
        }

        allocateBuffer(errcode_ret, usDev);
        executeNativeKernel(errcode_ret, usDev);

        usDev.clear();
    }

    public static void allocateBuffer(IntBuffer errcode_ret, UsefulDevice usDev) {

        long device = usDev.getDevice();
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

        try {
            // mem object destructor callbacks are called asynchronously on Nvidia
            destructorLatch.await();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        subbufferCB.free();

        bufferCB2.free();
        bufferCB1.free();
    }

    public static void executeNativeKernel(IntBuffer errcode_ret, UsefulDevice usDev) {

        long device = usDev.getDevice();
        long context = usDev.getContext();

        int errcode;

        long exec_caps = getDeviceInfoLong(device, CL_DEVICE_EXECUTION_CAPABILITIES);
        if ((exec_caps & CL_EXEC_NATIVE_KERNEL) != CL_EXEC_NATIVE_KERNEL) {
            System.out.println("No native kernel caps.");
            return;
        }

        System.out.println("\t\t-TRYING TO EXEC NATIVE KERNEL-");
        long queue = clCreateCommandQueue(context, device, NULL, errcode_ret);

        PointerBuffer ev = BufferUtils.createPointerBuffer(1);

        ByteBuffer kernelArgs = BufferUtils.createByteBuffer(4);
        kernelArgs.putInt(0, 1337);

        CLNativeKernel kernel;
        errcode = clEnqueueNativeKernel(queue, kernel = CLNativeKernel.create(
                args -> System.out.println("\t\tKERNEL EXEC argument: " + memByteBuffer(args, 4).getInt(0) + ", should be 1337")
        ), kernelArgs, null, null, null, ev);
        checkCLError(errcode);

        long e = ev.get(0);

        CountDownLatch latch = new CountDownLatch(1);

        CLEventCallback eventCB;
        errcode = clSetEventCallback(e, CL_COMPLETE, eventCB = CLEventCallback.create((event, event_command_exec_status, user_data) -> {
            System.out.println("\t\tEvent callback status: " + getEventStatusName(event_command_exec_status));
            latch.countDown();
        }), NULL);
        checkCLError(errcode);

        try {
            boolean expired = !latch.await(500, TimeUnit.MILLISECONDS);
            if (expired) {
                System.out.println("\t\tKERNEL EXEC FAILED!");
            }
        } catch (InterruptedException exc) {
            exc.printStackTrace();
        }
        eventCB.free();

        errcode = clReleaseEvent(e);
        checkCLError(errcode);
        kernel.free();

        kernelArgs = BufferUtils.createByteBuffer(POINTER_SIZE * 2);

        kernel = CLNativeKernel.create(args -> {
        });

        long time = System.nanoTime();
        int REPEAT = 1000;
        for (int i = 0; i < REPEAT; i++) {
            clEnqueueNativeKernel(queue, kernel, kernelArgs, null, null, null, null);
        }
        clFinish(queue);
        time = System.nanoTime() - time;

        System.out.printf("\n\t\tEMPTY NATIVE KERNEL AVG EXEC TIME: %.4fus\n", (double) time / (REPEAT * 1000));

        errcode = clReleaseCommandQueue(queue);
        checkCLError(errcode);
        kernel.free();

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
