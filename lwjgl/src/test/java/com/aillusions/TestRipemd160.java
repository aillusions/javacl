package com.aillusions;

import com.aillusions.cl.Main;
import junit.framework.TestCase;

/**
 * @author aillusions
 */
public class TestRipemd160 extends TestCase {

    public void testGetRipemd160() {
        assertEquals("a0b0d60e5991578ed37cbda2b17d8b2ce23ab295", Main.Ripemd160ToString(Main.getRipemd160("1FeexV6bAHb8ybZjqQMjJrcCrHGW9sb6uF")));
        assertEquals("f0cfda17f6fab20243958cbfb0fbe70fa929af92", Main.Ripemd160ToString(Main.getRipemd160("1NxJAqyjjAAV89cCSueVv4cfe797yyqFy1")));
    }
}
