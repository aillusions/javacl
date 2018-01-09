package com.aillusions.cl;

import com.aillusions.cl.demo.InfoUtil;
import com.aillusions.cl.device.OpenCLDeviceProvider;
import com.aillusions.cl.device.UsefulDevice;
import com.aillusions.cl.kernel.WindUpKernel;
import com.aillusions.cl.programm.LoadedProgram;
import com.aillusions.cl.programm.ProgramLoader;
import org.bitcoinj.core.Address;
import org.bitcoinj.core.ECKey;
import org.bitcoinj.core.NetworkParameters;
import org.bitcoinj.params.MainNetParams;
import org.lwjgl.BufferUtils;
import org.lwjgl.PointerBuffer;
import org.lwjgl.system.MemoryStack;
import org.spongycastle.crypto.params.ECDomainParameters;
import org.spongycastle.math.ec.ECCurve;
import org.spongycastle.math.ec.ECPoint;

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

    public static final List<String> real_patterns = Arrays.asList(
            "18iBkXFP5Ep7utstCkJYSUtXLUA49QbD5Q", // 84910071079903469711540322182995519010752598753241787550350295165653577042301
            "12NEsPS2tPhjXJHd3kGkTvQ7ECGypuxbeo", // 85373582762808404920801888792437094602169475096082456154754419692323304989563
            "1Em6NM1R4ZLPzfSvSapbVrA4CqbJduqg2C" // 86385075817194309241889933189838769976076542292920476979308177169247389148514
    );

    // Real addresses
    public static final List<String> patterns = new LinkedList<>();


    // static int patNumMax = 35_000_000; // Mac
    // static int patNumMax = 1_400_000_000; // PC
    static {
        System.out.println("Composing patterns list: in.");

        for (int i = 0; i < 1; i++) {
            patterns.addAll(real_patterns);
        }

        System.out.println("Total patterns: " + patterns.size());
    }


    static final int ACCESS_BUNDLE = 1024;
    static final int ACCESS_STRIDE = ACCESS_BUNDLE / 8;

    static final int hash160bytes = 20;
    static int patNum = patterns.size();

    static int target_table_buff_size = hash160bytes * patNum;
    static int response_buff_size = 28; //??

    static int nrows = 2048;
    static int ncols = 2560;
    static int round = nrows * ncols; // 5242880

    static final int nrows_SUM_ncols = nrows + ncols;

    static int z_heap_and_point_spaces = round_up_pow2(32 * 2 * round, 4096); // 335544320
    static int row_in_size = round_up_pow2(32 * 2 * ncols, 4096); // 163840

    static int invsize = 256;

    static final ECDomainParameters ecp = ECKey.CURVE;
    static final ECCurve curve = ecp.getCurve();
    static final ECPoint pGenConst = ecp.getG();
    static ECPoint pGenConstCopy = curve.createPoint(pGenConst.getAffineXCoord().toBigInteger(), pGenConst.getAffineYCoord().toBigInteger());

    static ECPoint pBatchInc;
    static ECPoint pOffset;

    static BigInteger vxc_bntmp;

    static {
        vxc_bntmp = BigInteger.valueOf(ncols);
        pBatchInc = pGenConst.multiply(vxc_bntmp);
        pBatchInc.normalize();

        vxc_bntmp = BigInteger.valueOf(round);
        pOffset = pGenConst.multiply(vxc_bntmp);
        pOffset.normalize();
    }

    static final ECPoint[] ppbase = new ECPoint[nrows_SUM_ncols];


    /**
     * Mac:
     * kernel_0: 210 ms.
     * kernel_1: 70 ms.
     * kernel_2: 2000 ms.
     * <p>
     * PC:
     * kernel_0: 50 ms.
     * kernel_1: 20 ms.
     * kernel_2: 70 ms.
     */
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

        setResultInitial(clQueue, kernel_2, errcode_ret);

        setPatterns(clQueue, kernel_2, errcode_ret);

        fillSeqPoints(clQueue, kernel_0, errcode_ret);
        rowIncrementTable();

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

        // mainLoop(clQueue, kernel_2, errcode_ret);

        program.clear();
        usDev.clear();
    }

    /* Set the found indicator for each slot -1 */
    public static void setResultInitial(long clQueue, WindUpKernel kernel_2, IntBuffer errcode_ret) {
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
            ocl_found_out.put((byte) i);
        }
        ocl_found_out.rewind();

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

    /* Write range records */
    public static void setPatterns(long clQueue, WindUpKernel kernel_2, IntBuffer errcode_ret) {

        System.out.println("setPatterns: in.");

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

        for (String pattern : patterns) {
            byte has160[] = getRipemd160(pattern);
            for (byte b : has160) {
                ocl_targets_in.put(b);
            }
        }

        ocl_targets_in.rewind();

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

        System.out.println("setPatterns: done.");
    }

    /* Fill the sequential point array */
    public static void fillSeqPoints(long clQueue, WindUpKernel kernel_0, IntBuffer errcode_ret) {

        System.out.println("fillSeqPoints: in.");

        {
            ECKey pRandKey = new ECKey();

            /* Build the base array of sequential points */

            ppbase[0] = pRandKey.getPubKeyPoint();

            for (int i = 1; i < ncols; i++) {
                ppbase[i] = pGenConst.add(ppbase[i - 1]);
            }

            curve.normalizeAll(ppbase);
        }

        long bufferArg = kernel_0.getBuffers()[2];

        ByteBuffer ocl_points_in = clEnqueueMapBuffer(
                clQueue,
                bufferArg,
                true,
                CL_MAP_WRITE,
                0L,
                row_in_size,
                null,
                null,
                errcode_ret,
                null
        );
        checkCLError(errcode_ret);

        for (int i = 0; i < ncols; i++) {
            vg_ocl_put_point_tpa(ocl_points_in, i, ppbase[i]);
        }

        ocl_points_in.rewind();

        PointerBuffer eventOut = BufferUtils.createPointerBuffer(1);
        int ret = clEnqueueUnmapMemObject(
                clQueue,
                bufferArg,
                ocl_points_in,
                null,
                eventOut
        );
        checkCLError(ret);

        clWaitForEvents(eventOut);
        clReleaseEvent(eventOut.get());

        System.out.println("fillSeqPoints: done.");
    }

    /* row increment table. */
    public static void rowIncrementTable() {
        int idxFrom = ncols;

        ppbase[idxFrom] = pGenConstCopy;
        for (int i = 1; i < nrows; i++) {
            ppbase[idxFrom + i] = pBatchInc.add(ppbase[idxFrom + i - 1]);
        }

        curve.normalizeAll(ppbase);
    }

    static void mainLoop(long clQueue, WindUpKernel kernel_2, IntBuffer errcode_ret) {

        int rekey_at = 100000000;
        int npoints = 1;
        int c = 0;

        boolean slot_busy = false;
        boolean slot_done = false;

        while (true) {
            if (slot_done) {
                boolean rv = checkResult(clQueue, kernel_2, errcode_ret);
                if (rv) {
                    System.out.println("Found?");
                    return;
                }

                c += round;
            }

            if ((npoints + round) < rekey_at) {
                if (npoints > 1) {
                    /* Move the row increments forward */
                    for (int i = 0; i < nrows; i++) {

                        int idxFrom = ncols;

                        ppbase[idxFrom] = pGenConstCopy;
                        for (int j = 1; j < nrows; j++) {
                            ppbase[idxFrom + j] = pBatchInc.add(ppbase[idxFrom + j - 1]);
                        }

                        curve.normalizeAll(ppbase);
                    }

                }

                // TODO Copy the row stride array to the device

                npoints += round;
            }

            slot_done = true;
        }
    }

    static void vg_ocl_put_point_tpa(ByteBuffer buf, int cell, ECPoint ppnt) {

        //ByteBuffer pntbuf = ByteBuffer.allocate(64);
        ByteBuffer pntbuf = BufferUtils.createByteBuffer(64);

        vg_ocl_put_point(pntbuf, ppnt);

        int start = ((((2 * cell) / ACCESS_STRIDE) * ACCESS_BUNDLE) + (cell % (ACCESS_STRIDE / 2)));

        for (int i = 0; i < 8; i++) {
            int copyToOffset = 4 * (start + i * ACCESS_STRIDE); // 0, 512, 1024, 1536, 2048, 2560, 3072, 3584
            int copyFromOffset = (i * 4); // 0, 4, 8, 12, 16, 20, 24, 28
            final int copySize = 4;

            copyBuff(buf, pntbuf, copyToOffset, copyFromOffset, copySize);
        }

        for (int i = 0; i < 8; i++) {
            int copyToOffset = 4 * (start + (ACCESS_STRIDE / 2) + (i * ACCESS_STRIDE)); // 256, 768, 1280, 1792, 2304, 2816, 3328, 3840, 260, 772, 1284..
            int copyFromOffset = 32 + (i * 4); // 32, 36, 40, 44, 48, 52, 56, 60, 32, 36, 40 ..
            final int copySize = 4;
            copyBuff(buf, pntbuf, copyToOffset, copyFromOffset, copySize);
        }
    }

    static void copyBuff(ByteBuffer to, ByteBuffer from, int toOffset, int fromOffset, int length) {
        for (int i = 0; i < length; i++) {
            int toIdx = toOffset + i;
            int fromIdx = fromOffset + i;
            to.put(toIdx, from.get(fromIdx));
        }
    }

    static void vg_ocl_put_point(ByteBuffer buf, ECPoint ppnt) {
        assert (ppnt.isNormalized());
        vg_ocl_put_bignum_raw(buf, ppnt.getAffineXCoord().toBigInteger());
        vg_ocl_put_bignum_raw(buf, ppnt.getAffineYCoord().toBigInteger());
    }

    static void vg_ocl_put_bignum_raw(ByteBuffer buf, BigInteger bn) {
        byte[] bytes = bn.toByteArray();
        int bnlen = bytes.length;

        // System.out.println("vg_ocl_put_bignum_raw: adding: " + bnlen + " bytes.");

        if (bnlen >= 32) {
            buf.put(bytes, 0, 32);
        } else {
            buf.put(bytes, 0, bnlen);
            buf.put(new byte[32 - bnlen]);
        }
    }

    public static boolean checkResult(long clQueue, WindUpKernel kernel_2, IntBuffer errcode_ret) {
        long bufferArg = kernel_2.getBuffers()[0];

        ByteBuffer ocl_found_out = clEnqueueMapBuffer(
                clQueue,
                bufferArg,
                true,
                CL_MAP_READ | CL_MAP_WRITE,
                0L,
                response_buff_size,
                null,
                null,
                errcode_ret,
                null
        );
        checkCLError(errcode_ret);

        byte[] bytes = new byte[response_buff_size];
        for (int i = 0; i < response_buff_size; i++) {
            bytes[i] = ocl_found_out.get(i);
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

        return false;
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
            int sizeof = z_heap_and_point_spaces;
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
            int sizeof = z_heap_and_point_spaces;
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
            int sizeof = row_in_size;
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
            int sizeof = z_heap_and_point_spaces;
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
            int sizeof = z_heap_and_point_spaces;
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
            int sizeof = z_heap_and_point_spaces;
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

    // Private key hexadecimal - 256 bits in hexadecimal is 32 bytes
    // Specifically, any 256-bit number from 0x1 to 0xFFFF FFFF FFFF FFFF FFFF FFFF FFFF FFFE BAAE DCE6 AF48 A03B BFD2 5E8C D036 4140 is a valid private key.

}

//ByteBuffer pntbuf = stack.malloc(64);
//IntBuffer buffer0 = stack.callocInt(1);
//ByteBuffer col_in = BufferUtils.createByteBuffer();
//clRetainMemObject(col_in);
//clSetKernelArg(clKernel, 3, col_in);

//ByteBuffer mapped_ptr = BufferUtils.createByteBuffer(target_table_buff_size);
/*ByteBuffer mapped_ptr = stack.malloc(target_table_buff_size);
for (int i = 0; i < mapped_ptr.capacity(); i++) {
    mapped_ptr.put(i, (byte) i);
}*/

/*

      X9ECParameters ecSpec = SECNamedCurves.getByName("secp256k1");
        ECCurve.Fp eccurve = (ECCurve.Fp) ecSpec.getCurve();
        ECPoint G = ecSpec.getG();
        ECKey.CURVE.getN();
        ECCurve curveFromKey = pRandKey.getPubKeyPoint().getCurve(); // pgroup
        ECPoint gFromKey = ECKey.CURVE.getG(); // pgen

        //curveFromKey.

        // The private key is simply a BIGNUM, see BN_new(3).
        BigInteger privKey = pRandKey.getPrivKey();

        // The public key is a point on a curve represented by an EC_POINT
        ECPoint ecPoint = pRandKey.getPubKeyPoint();
        ECCurve ecCurve = ecPoint.getCurve();*/