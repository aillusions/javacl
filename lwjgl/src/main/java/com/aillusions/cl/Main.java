package com.aillusions.cl;

import com.aillusions.cl.device.OpenCLDeviceProvider;
import com.aillusions.cl.device.UsefulDevice;
import org.lwjgl.BufferUtils;
import org.lwjgl.PointerBuffer;
import org.lwjgl.opencl.CL10;
import org.lwjgl.opencl.CLProgramCallback;
import org.lwjgl.system.Configuration;
import org.lwjgl.system.MemoryStack;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.concurrent.CountDownLatch;

import static com.aillusions.cl.demo.IOUtil.ioResourceToByteBuffer;
import static com.aillusions.cl.demo.InfoUtil.*;
import static org.lwjgl.opencl.CL10.*;
import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.system.MemoryUtil.NULL;

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

    private static void demo(MemoryStack stack) throws IOException {

        IntBuffer errcode_ret = stack.callocInt(1);

        UsefulDevice usDev = OpenCLDeviceProvider.getUsefulDevice(stack, errcode_ret);
        if (usDev == null) {
            System.out.println("No useful device found.");
            return;
        } else {
            System.out.println("Useful device: " + usDev.toString());
        }

        executeKernel(errcode_ret, usDev);

        usDev.clear();
    }

    public static void executeKernel(IntBuffer errcode_ret, UsefulDevice usDev) throws IOException {

        int n = 10;

        float srcArrayA[] = new float[n];
        float srcArrayB[] = new float[n];
        float dstArray[] = new float[n];

        for (int i = 0; i < n; i++) {
            srcArrayA[i] = i;
            srcArrayB[i] = i * 2;
        }

        long device = usDev.getDevice();
        long context = usDev.getContext();

        PointerBuffer strings = BufferUtils.createPointerBuffer(1);
        PointerBuffer lengths = BufferUtils.createPointerBuffer(1);

        ByteBuffer source = ioResourceToByteBuffer("OpenCLSum.cl", 4096);
        strings.put(0, source);
        lengths.put(0, source.remaining());

        long clProgram = clCreateProgramWithSource(context, strings, lengths, errcode_ret);
        checkCLError(errcode_ret);

        StringBuilder options = new StringBuilder("");

        CountDownLatch latch = new CountDownLatch(1);

        CLProgramCallback buildCallback;
        int errcode = clBuildProgram(clProgram, device, options, buildCallback = CLProgramCallback.create((program, user_data) -> {
            System.out.println(String.format(
                    "The cl_program [0x%X] was built %s",
                    program,
                    getProgramBuildInfoInt(program, device, CL_PROGRAM_BUILD_STATUS) == CL_SUCCESS ? "successfully" : "unsuccessfully"
            ));
            String log = getProgramBuildInfoStringASCII(program, device, CL_PROGRAM_BUILD_LOG);
            if (!log.isEmpty()) {
                System.out.println(String.format("BUILD LOG:\n----\n%s\n-----", log));
            }

            latch.countDown();
        }), NULL);
        checkCLError(errcode);

        try {
            latch.await();
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }

        buildCallback.free();

        // init kernel with constants
        long clKernel = clCreateKernel(clProgram, "openCLSumK", errcode_ret);
        checkCLError(errcode_ret);

        long bufferArg1 = allocateBufferFor(errcode_ret, usDev, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, srcArrayA);
        long bufferArg2 = allocateBufferFor(errcode_ret, usDev, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, srcArrayB);
        long bufferArg3 = allocateBufferFor(errcode_ret, usDev, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, dstArray);

        clSetKernelArg1p(clKernel, 0, bufferArg1);
        clSetKernelArg1p(clKernel, 1, bufferArg2);
        clSetKernelArg1p(clKernel, 2, bufferArg3);

        long clQueue = clCreateCommandQueue(usDev.getContext(), device, 0, errcode_ret);

        PointerBuffer globalWorkSize = BufferUtils.createPointerBuffer(1);
        globalWorkSize.put(0, n);

        errcode = clEnqueueNDRangeKernel(
                clQueue,
                clKernel,
                1,
                null,
                globalWorkSize,
                null,
                null,
                null);
        checkCLError(errcode);

        CL10.clFinish(clQueue);

        FloatBuffer resultBuff = BufferUtils.createFloatBuffer(n);
        CL10.clEnqueueReadBuffer(clQueue, bufferArg3, true, 0L, resultBuff, null, null);

        for (int i = 0; i < resultBuff.capacity(); i++) {
            System.out.println(resultBuff.get(i));
        }

        // Destroy our kernel and program
        CL10.clReleaseKernel(clKernel);
        CL10.clReleaseProgram(clProgram);
        // Destroy our memory objects
        CL10.clReleaseMemObject(bufferArg1);
        CL10.clReleaseMemObject(bufferArg2);
        CL10.clReleaseMemObject(bufferArg3);
    }

    public static long allocateBufferFor(IntBuffer errcode_ret, UsefulDevice usDev, long flags, float[] dataArray) {

        long context = usDev.getContext();

        FloatBuffer dataBuff = BufferUtils.createFloatBuffer(dataArray.length);
        dataBuff.put(dataArray);
        dataBuff.rewind();

        long buffer = clCreateBuffer(context, flags, dataBuff, errcode_ret);
        checkCLError(errcode_ret);

        return buffer;
    }

}
