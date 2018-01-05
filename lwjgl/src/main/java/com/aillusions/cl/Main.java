package com.aillusions.cl;

import org.lwjgl.opencl.CL;
import org.lwjgl.system.Configuration;

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

    public static void main(String... args) {

        try {
            try {
                CL.create();
            } catch (Exception e) {
                System.out.println("CL init nok: " + e.getMessage());
            }




        } finally {
            CL.destroy();
        }
    }
}
