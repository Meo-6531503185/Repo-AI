

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;



public class HelloWorldTest {
    @Test
    void testGreet() {
        assertEquals("Hello, World!", HelloWorld.greet());
    }
}

