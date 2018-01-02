package com.aillusions;

import org.apache.commons.io.IOUtils;

import java.io.IOException;

/**
 * @author aillusions
 */
public class ResourcesIO {

    public static String getCode() throws IOException {
        return IOUtils.toString(ResourcesIO.class.getClassLoader().getResourceAsStream("calc_addrs.cl"), "UTF-8");
    }
}
