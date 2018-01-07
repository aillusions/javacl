package com.aillusions.cl;

import com.aillusions.cl.demo.InfoUtil;
import com.aillusions.cl.device.OpenCLDeviceProvider;
import com.aillusions.cl.device.UsefulDevice;
import com.aillusions.cl.programm.LoadedProgram;
import com.aillusions.cl.programm.ProgramLoader;
import org.lwjgl.BufferUtils;
import org.lwjgl.PointerBuffer;
import org.lwjgl.system.MemoryStack;

import java.io.IOException;
import java.nio.IntBuffer;

import static com.aillusions.cl.demo.InfoUtil.checkCLError;
import static com.aillusions.cl.programm.ProgramLoader.getFileNamePath;
import static org.lwjgl.opencl.CL10.*;
import static org.lwjgl.system.MemoryStack.stackPush;

/**
 * @author aillusions
 */
public class Main {

    private static final String ec_add_grid = "ec_add_grid"; // [0]
    private static final String heap_invert = "heap_invert"; // [1]
    private static final String hash_ec_point_search_prefix = "hash_ec_point_search_prefix"; // [2]

    private static final String calc_addr_file = getFileNamePath("calc_addrs");

    private static final int n = 10;

    private static final float anArray[] = new float[n];

    static {
        for (int i = 0; i < n; i++) {
            anArray[i] = i;
        }
    }

    /*public static boolean is_pow2(int v) {
        return !(v & (v - 1));
    }*/

    public static int round_up_pow2(int x, int a) {
        return (((x) + ((a) - 1)) & ~((a) - 1));
    }

    static int patNum = 2;
    static int nrows = 2048;
    static int ncols = 2560;
    static int round = nrows * ncols;
    static int invsize = 256;

    public static void main(String... arg) throws IOException {

        MemoryStack stack = stackPush();
        IntBuffer errcode_ret = stack.callocInt(1);

        UsefulDevice usDev = OpenCLDeviceProvider.getUsefulDevice(stack, errcode_ret);
        if (usDev == null) {
            System.out.println("No useful device found.");
            return;
        } else {
            System.out.println("Useful device: " + usDev.toString());
        }

        LoadedProgram program = ProgramLoader.loadProgramm(
                errcode_ret,
                usDev,
                calc_addr_file,
                ec_add_grid,
                heap_invert,
                // "hash_ec_point_get", // ??
                hash_ec_point_search_prefix);


        long clQueue = clCreateCommandQueue(usDev.getContext(), usDev.getDevice(), 0, errcode_ret);

        PointerBuffer globalws = BufferUtils.createPointerBuffer(2);
        globalws.put(0, ncols);
        globalws.put(1, nrows);

        PointerBuffer invws = BufferUtils.createPointerBuffer(1);
        invws.put(0, (ncols * nrows) / invsize); // 20480 ?

        long kernel_0 = initKernel0(usDev, program, errcode_ret);
        long kernel_1 = initKernel1(usDev, program, errcode_ret);
        long kernel_2 = initKernel2(usDev, program, errcode_ret);

        /*
            Mac:
                kernel_0: 200 ms.
                kernel_1: 70 ms.
                kernel_2: 1000 ms.
         */

        /*
            PC:
                kernel_0: 50 ms.
                kernel_1: 20 ms.
                kernel_2: 70 ms.
         */
        {
            long start = System.currentTimeMillis();
            enqueueAndWait(clQueue, kernel_0, 2, globalws);
            System.out.println("kernel_0: " + (System.currentTimeMillis() - start) + " ms.");
        }

        {
            long start = System.currentTimeMillis();
            enqueueAndWait(clQueue, kernel_1, 1, invws);
            System.out.println("kernel_1: " + (System.currentTimeMillis() - start) + " ms.");
        }

        {
            long start = System.currentTimeMillis();
            enqueueAndWait(clQueue, kernel_2, 2, globalws);
            System.out.println("kernel_2: " + (System.currentTimeMillis() - start) + " ms.");
        }

        program.clear();
        usDev.clear();
    }

    public static void enqueueAndWait(long clQueue, long kernel, int dimm, PointerBuffer ws) {

        PointerBuffer eventOut = BufferUtils.createPointerBuffer(1);

        int errcode = clEnqueueNDRangeKernel(
                clQueue,
                kernel,
                dimm,
                null,
                ws,
                null,
                null,
                eventOut);
        checkCLError(errcode);

        eventOut.rewind();
        clWaitForEvents(eventOut);
        long eventAddr = eventOut.get(0);

        errcode = clReleaseEvent(eventAddr);
        InfoUtil.checkCLError(errcode);
    }

    /**
     * 0
     */
    public static long initKernel0(UsefulDevice usDev,
                                   LoadedProgram program,
                                   IntBuffer errcode_ret) {

        long clKernel = program.getKernel(ec_add_grid);

        {
            int sizeof = round_up_pow2(32 * 2 * round, 4096); // 335544320 ?
            long points_out = clCreateBuffer(usDev.getContext(), CL_MEM_READ_WRITE, sizeof, errcode_ret);
            clRetainMemObject(points_out);
            PointerBuffer mem_list = BufferUtils.createPointerBuffer(1);
            mem_list.put(0, points_out);
            clSetKernelArg(clKernel, 0, mem_list);
        }

        {
            int sizeof = round_up_pow2(32 * 2 * round, 4096); // 335544320 ?
            long z_heap = clCreateBuffer(usDev.getContext(), CL_MEM_READ_WRITE, sizeof, errcode_ret);
            clRetainMemObject(z_heap);
            PointerBuffer mem_list = BufferUtils.createPointerBuffer(1);
            mem_list.put(0, z_heap);
            clSetKernelArg(clKernel, 1, mem_list);
        }

        {
            int sizeof = round_up_pow2(32 * 2 * ncols, 4096); // 163840 ?
            long row_in = clCreateBuffer(usDev.getContext(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof, errcode_ret);
            clRetainMemObject(row_in);
            PointerBuffer mem_list = BufferUtils.createPointerBuffer(1);
            mem_list.put(0, row_in);
            clSetKernelArg(clKernel, 2, mem_list);
        }

        {
            int sizeof = 32 * 2 * nrows; // ?
            long col_in = clCreateBuffer(usDev.getContext(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof, errcode_ret);
            clRetainMemObject(col_in);
            PointerBuffer mem_list = BufferUtils.createPointerBuffer(1);
            mem_list.put(0, col_in);
            clSetKernelArg(clKernel, 3, mem_list);
        }

        return clKernel;
    }

    /**
     * 1
     */
    public static long initKernel1(UsefulDevice usDev,
                                   LoadedProgram program,
                                   IntBuffer errcode_ret) {

        long clKernel = program.getKernel(heap_invert);

        {
            int sizeof = round_up_pow2(32 * 2 * round, 4096); // 335544320 ?
            long z_heap = clCreateBuffer(usDev.getContext(), CL_MEM_READ_WRITE, sizeof, errcode_ret);
            clRetainMemObject(z_heap);
            PointerBuffer mem_list = BufferUtils.createPointerBuffer(1);
            mem_list.put(0, z_heap);
            clSetKernelArg(clKernel, 0, mem_list);
        }

        {
            int batch = 256;
            clSetKernelArg1i(clKernel, 1, batch);
        }

        return clKernel;
    }

    /**
     * 2
     */
    public static long initKernel2(UsefulDevice usDev,
                                   LoadedProgram program,
                                   IntBuffer errcode_ret) {

        long clKernel = program.getKernel(hash_ec_point_search_prefix);

        {
            PointerBuffer found = BufferUtils.createPointerBuffer(1);
            clSetKernelArg(clKernel, 0, found);
        }

        {
            int sizeof = round_up_pow2(32 * 2 * round, 4096); // 335544320 ?
            long points_in = clCreateBuffer(usDev.getContext(), CL_MEM_READ_WRITE, sizeof, errcode_ret);
            clRetainMemObject(points_in);
            PointerBuffer mem_list = BufferUtils.createPointerBuffer(1);
            mem_list.put(0, points_in);
            clSetKernelArg(clKernel, 1, mem_list);
        }

        {
            int sizeof = round_up_pow2(32 * 2 * round, 4096); // 335544320 ?
            long z_heap = clCreateBuffer(usDev.getContext(), CL_MEM_READ_WRITE, sizeof, errcode_ret);
            clRetainMemObject(z_heap);
            PointerBuffer mem_list = BufferUtils.createPointerBuffer(1);
            mem_list.put(0, z_heap);
            clSetKernelArg(clKernel, 2, mem_list);
        }

        {
            int sizeof = 40 * patNum;
            long target_table = clCreateBuffer(usDev.getContext(), CL_MEM_READ_WRITE, sizeof, errcode_ret);
            clRetainMemObject(target_table);
            PointerBuffer mem_list = BufferUtils.createPointerBuffer(1);
            mem_list.put(0, target_table);
            clSetKernelArg(clKernel, 3, mem_list);
        }

        int ntargets = patNum;
        clSetKernelArg1i(clKernel, 4, ntargets);

        return clKernel;
    }

}

//IntBuffer buffer0 = stack.callocInt(1);
//ByteBuffer col_in = BufferUtils.createByteBuffer();
//clRetainMemObject(col_in);
//clSetKernelArg(clKernel, 3, col_in);