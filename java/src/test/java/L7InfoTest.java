import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class L7InfoTest {
    @Test
    void testGetPrice() {
        L7Info l7Info = new L7Info(null);
        assertEquals(50000, l7Info.getPrice());
    }

    @Test
    void testGetBill() {
        L7Info l7Info = new L7Info(null);
        assertEquals("Pay per unit", l7Info.getBill());
    }
}
