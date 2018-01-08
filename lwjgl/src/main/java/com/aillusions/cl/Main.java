package com.aillusions.cl;

import com.aillusions.cl.demo.InfoUtil;
import com.aillusions.cl.device.OpenCLDeviceProvider;
import com.aillusions.cl.device.UsefulDevice;
import com.aillusions.cl.kernel.WindUpKernel;
import com.aillusions.cl.programm.LoadedProgram;
import com.aillusions.cl.programm.ProgramLoader;
import org.bitcoinj.core.Address;
import org.bitcoinj.core.NetworkParameters;
import org.bitcoinj.params.MainNetParams;
import org.lwjgl.BufferUtils;
import org.lwjgl.PointerBuffer;
import org.lwjgl.system.MemoryStack;

import java.io.IOException;
import java.math.BigInteger;
import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

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

    /*
        available on the device:
            Mac: 2130706432 bytes (2Gb)
    */

    // Real addresses
    public static List<String> patterns = new LinkedList<>(Arrays.asList(
            "18iBkXFP5Ep7utstCkJYSUtXLUA49QbD5Q", // 84910071079903469711540322182995519010752598753241787550350295165653577042301
            "12NEsPS2tPhjXJHd3kGkTvQ7ECGypuxbeo", // 85373582762808404920801888792437094602169475096082456154754419692323304989563
            "1Em6NM1R4ZLPzfSvSapbVrA4CqbJduqg2C") // 86385075817194309241889933189838769976076542292920476979308177169247389148514
    );


    static final int hash160bytes = 20;

    static int patNum = patterns.size();

    // static int patNumMax = 35_000_000; // Mac
    // static int patNumMax = 1_400_000_000; // PC

    static int target_table_buff_size = hash160bytes * patNum;
    static int response_buff_size = hash160bytes + 8;

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

        WindUpKernel kernel_0 = initKernel0(usDev, program, errcode_ret);
        WindUpKernel kernel_1 = initKernel1(usDev, program, errcode_ret);
        WindUpKernel kernel_2 = initKernel2(usDev, program, errcode_ret);

        /* Set the found indicator for each slot -1 */
        {

            long bufferArg = kernel_2.getBuffers()[0];
            ByteBuffer ocl_found_out = clEnqueueMapBuffer(
                    clQueue,
                    bufferArg,
                    true,
                    CL_MAP_WRITE,
                    0L,
                    response_buff_size,
                    null,
                    null,
                    errcode_ret,
                    null
            );
            checkCLError(errcode_ret);

            for (int i = 0; i < response_buff_size; i++) {
                ocl_found_out.put((byte) 0);
            }

            PointerBuffer eventOut = BufferUtils.createPointerBuffer(1);

            int ret = clEnqueueUnmapMemObject(
                    clQueue,
                    bufferArg,
                    ocl_found_out,
                    null,
                    eventOut
            );
            checkCLError(ret);

            clWaitForEvents(eventOut);
            clReleaseEvent(eventOut.get());
        }

        {   /* Write range records */

            long bufferArg = kernel_2.getBuffers()[3];

            ByteBuffer ocl_targets_in = clEnqueueMapBuffer(
                    clQueue,
                    bufferArg,
                    true,
                    CL_MAP_WRITE,
                    0L,
                    target_table_buff_size, // the size in bytes of the region in the buffer object that is being mapped
                    null,
                    null,
                    errcode_ret,
                    null
            );
            checkCLError(errcode_ret);

            // TODO populate patterns table here

            for (String pattern : patterns) {
                byte has160[] = getRipemd160(pattern);
                for (byte b : has160) {
                    ocl_targets_in.put(b);
                }
            }

            PointerBuffer eventOut = BufferUtils.createPointerBuffer(1);
            int ret = clEnqueueUnmapMemObject(
                    clQueue,
                    bufferArg,
                    ocl_targets_in,
                    null,
                    eventOut
            );
            checkCLError(ret);

            clWaitForEvents(eventOut);
            clReleaseEvent(eventOut.get());
        }

        /*
            Mac:
                kernel_0: 210 ms.
                kernel_1: 70 ms.
                kernel_2: 2000 ms.
         */

        /*
            PC:
                kernel_0: 50 ms.
                kernel_1: 20 ms.
                kernel_2: 70 ms.
         */
        {
            long start = System.currentTimeMillis();
            enqueueAndWait(clQueue, kernel_0.getKernel(), 2, globalws);
            System.out.println("kernel_0: " + (System.currentTimeMillis() - start) + " ms.");
        }

        {
            long start = System.currentTimeMillis();
            enqueueAndWait(clQueue, kernel_1.getKernel(), 1, invws);
            System.out.println("kernel_1: " + (System.currentTimeMillis() - start) + " ms.");
        }

        {
            long start = System.currentTimeMillis();
            enqueueAndWait(clQueue, kernel_2.getKernel(), 2, globalws);
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
    public static WindUpKernel initKernel0(UsefulDevice usDev,
                                           LoadedProgram program,
                                           IntBuffer errcode_ret) {

        long clKernel = program.getKernel(ec_add_grid);
        long[] buffers = new long[4];

        {
            int argIdx = 0;
            int sizeof = round_up_pow2(32 * 2 * round, 4096); // 335544320 ?
            long points_out = clCreateBuffer(usDev.getContext(), CL_MEM_READ_WRITE, sizeof, errcode_ret);
            checkCLError(errcode_ret);

            checkCLError(clRetainMemObject(points_out));

            PointerBuffer mem_list = BufferUtils.createPointerBuffer(1);
            mem_list.put(0, points_out);
            clSetKernelArg(clKernel, argIdx, mem_list);
            buffers[argIdx] = points_out;
        }

        {
            int argIdx = 1;
            int sizeof = round_up_pow2(32 * 2 * round, 4096); // 335544320 ?
            long z_heap = clCreateBuffer(usDev.getContext(), CL_MEM_READ_WRITE, sizeof, errcode_ret);
            checkCLError(errcode_ret);

            checkCLError(clRetainMemObject(z_heap));

            PointerBuffer mem_list = BufferUtils.createPointerBuffer(1);
            mem_list.put(0, z_heap);
            clSetKernelArg(clKernel, argIdx, mem_list);
            buffers[argIdx] = z_heap;
        }

        {
            int argIdx = 2;
            int sizeof = round_up_pow2(32 * 2 * ncols, 4096); // 163840 ?
            long row_in = clCreateBuffer(usDev.getContext(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof, errcode_ret);
            checkCLError(errcode_ret);

            checkCLError(clRetainMemObject(row_in));

            PointerBuffer mem_list = BufferUtils.createPointerBuffer(1);
            mem_list.put(0, row_in);
            clSetKernelArg(clKernel, argIdx, mem_list);
            buffers[argIdx] = row_in;
        }

        {
            int argIdx = 3;
            int sizeof = 32 * 2 * nrows; // ?
            long col_in = clCreateBuffer(usDev.getContext(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof, errcode_ret);
            checkCLError(errcode_ret);

            checkCLError(clRetainMemObject(col_in));

            PointerBuffer mem_list = BufferUtils.createPointerBuffer(1);
            mem_list.put(0, col_in);
            clSetKernelArg(clKernel, argIdx, mem_list);
            buffers[argIdx] = col_in;
        }

        return new WindUpKernel(clKernel, buffers);
    }

    /**
     * 1
     */
    public static WindUpKernel initKernel1(UsefulDevice usDev,
                                           LoadedProgram program,
                                           IntBuffer errcode_ret) {

        long clKernel = program.getKernel(heap_invert);
        long[] buffers = new long[2];

        {
            int argIdx = 0;
            int sizeof = round_up_pow2(32 * 2 * round, 4096); // 335544320 ?
            long z_heap = clCreateBuffer(usDev.getContext(), CL_MEM_READ_WRITE, sizeof, errcode_ret);
            checkCLError(errcode_ret);

            checkCLError(clRetainMemObject(z_heap));

            PointerBuffer mem_list = BufferUtils.createPointerBuffer(1);
            mem_list.put(0, z_heap);
            clSetKernelArg(clKernel, argIdx, mem_list);
            buffers[argIdx] = z_heap;
        }

        {
            int argIdx = 1;
            int batch = 256;
            checkCLError(clSetKernelArg1i(clKernel, argIdx, batch));
            buffers[argIdx] = batch;
        }

        return new WindUpKernel(clKernel, buffers);
    }

    /**
     * 2
     */
    public static WindUpKernel initKernel2(UsefulDevice usDev,
                                           LoadedProgram program,
                                           IntBuffer errcode_ret) {

        long clKernel = program.getKernel(hash_ec_point_search_prefix);
        long[] buffers = new long[5];

        {
            int argIdx = 0;
            int sizeof = response_buff_size;
            long found = clCreateBuffer(usDev.getContext(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof, errcode_ret);
            checkCLError(errcode_ret);

            checkCLError(clRetainMemObject(found));

            PointerBuffer mem_list = BufferUtils.createPointerBuffer(1);
            mem_list.put(0, found);
            clSetKernelArg(clKernel, argIdx, mem_list);
            buffers[argIdx] = found;
        }

        {
            int argIdx = 1;
            int sizeof = round_up_pow2(32 * 2 * round, 4096); // 335544320 ?
            long points_in = clCreateBuffer(usDev.getContext(), CL_MEM_READ_WRITE, sizeof, errcode_ret);
            checkCLError(errcode_ret);

            checkCLError(clRetainMemObject(points_in));

            PointerBuffer mem_list = BufferUtils.createPointerBuffer(1);
            mem_list.put(0, points_in);
            clSetKernelArg(clKernel, argIdx, mem_list);
            buffers[argIdx] = points_in;
        }

        {
            int argIdx = 2;
            int sizeof = round_up_pow2(32 * 2 * round, 4096); // 335544320 ?
            long z_heap = clCreateBuffer(usDev.getContext(), CL_MEM_READ_WRITE, sizeof, errcode_ret);
            checkCLError(errcode_ret);

            checkCLError(clRetainMemObject(z_heap));

            PointerBuffer mem_list = BufferUtils.createPointerBuffer(1);
            mem_list.put(0, z_heap);
            clSetKernelArg(clKernel, argIdx, mem_list);
            buffers[argIdx] = z_heap;
        }

        {/* (re)allocate target buffer */
            int argIdx = 3;
            int sizeof = target_table_buff_size;
            long target_table = clCreateBuffer(usDev.getContext(), CL_MEM_READ_WRITE, sizeof, errcode_ret);
            checkCLError(errcode_ret);

            checkCLError(clRetainMemObject(target_table));

            PointerBuffer mem_list = BufferUtils.createPointerBuffer(1);
            mem_list.put(0, target_table);
            clSetKernelArg(clKernel, argIdx, mem_list);
            buffers[argIdx] = target_table;
        }

        {
            int argIdx = 4;
            int ntargets = patNum;
            checkCLError(clSetKernelArg1i(clKernel, argIdx, ntargets));
            buffers[argIdx] = ntargets;
        }

        return new WindUpKernel(clKernel, buffers);
    }

    //  Mainline addresses can be 25-34 characters in length. Most addresses are 33 or 34 characters long.
    // 25-byte binary Bitcoin Address.
    // 20 bytes -> 40 hex-digits
    // 4 bytes -> 8 hex-digits
    // The 160-bit RIPEMD-160 hashes (RIPE message digests) are typically represented as 40-digit hexadecimal numbers.
    public static byte[] getRipemd160(String base58Address) {
        final NetworkParameters netParams = MainNetParams.get();
        Address addr = Address.fromBase58(netParams, base58Address);
        byte[] hash160 = addr.getHash160();
        return hash160;

    }

    public static String Ripemd160ToString(byte[] hash160) {
        return new BigInteger(1, hash160).toString(16);
    }

}

//IntBuffer buffer0 = stack.callocInt(1);
//ByteBuffer col_in = BufferUtils.createByteBuffer();
//clRetainMemObject(col_in);
//clSetKernelArg(clKernel, 3, col_in);

//ByteBuffer mapped_ptr = BufferUtils.createByteBuffer(target_table_buff_size);
/*ByteBuffer mapped_ptr = stack.malloc(target_table_buff_size);
for (int i = 0; i < mapped_ptr.capacity(); i++) {
    mapped_ptr.put(i, (byte) i);
}*/