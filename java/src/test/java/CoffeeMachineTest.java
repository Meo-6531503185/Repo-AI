import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class CoffeeMachineTest {

    @Test
    void testInitialState() {
        CoffeeMachine machine = new CoffeeMachine(400, 540, 120, 9, 550, 20, 15, 10, 5);
        assertEquals(400, machine.getWater());
        assertEquals(540, machine.getMilk());
        assertEquals(120, machine.getCoffeeBeans());
        assertEquals(9, machine.getCups());
        assertEquals(550, machine.getMoney());
        assertEquals(20, machine.getCookies());
        assertEquals(15, machine.getBrownies());
        assertEquals(10, machine.getTarts());
        assertEquals(5, machine.getMochi());
    }

    @Test
    void testMakeCoffeeEspressoSuccess() {
        CoffeeMachine machine = new CoffeeMachine(400, 540, 120, 9, 550, 20, 15, 10, 5);
        machine.makeCoffee(250, 0, 16, 4);
        assertEquals(150, machine.getWater());
        assertEquals(540, machine.getMilk());
        assertEquals(104, machine.getCoffeeBeans());
        assertEquals(8, machine.getCups());
        assertEquals(554, machine.getMoney());
    }

    @Test
    void testMakeCoffeeLatteSuccess() {
        CoffeeMachine machine = new CoffeeMachine(400, 540, 120, 9, 550, 20, 15, 10, 5);
        machine.makeCoffee(350, 75, 20, 7);
        assertEquals(50, machine.getWater());
        assertEquals(465, machine.getMilk());
        assertEquals(100, machine.getCoffeeBeans());
        assertEquals(8, machine.getCups());
        assertEquals(557, machine.getMoney());
    }

    @Test
    void testMakeCoffeeCappuccinoSuccess() {
        CoffeeMachine machine = new CoffeeMachine(400, 540, 120, 9, 550, 20, 15, 10, 5);
        machine.makeCoffee(200, 100, 12, 6);
        assertEquals(200, machine.getWater());
        assertEquals(440, machine.getMilk());
        assertEquals(108, machine.getCoffeeBeans());
        assertEquals(8, machine.getCups());
        assertEquals(556, machine.getMoney());
    }

    @Test
    void testMakeCoffeeNotEnoughWater() {
        CoffeeMachine machine = new CoffeeMachine(100, 540, 120, 9, 550, 20, 15, 10, 5);
        machine.makeCoffee(250, 0, 16, 4);
        assertEquals(100, machine.getWater());
        assertEquals(540, machine.getMilk());
        assertEquals(120, machine.getCoffeeBeans());
        assertEquals(9, machine.getCups());
        assertEquals(550, machine.getMoney());
    }

    @Test
    void testMakeCoffeeNotEnoughMilk() {
        CoffeeMachine machine = new CoffeeMachine(400, 50, 120, 9, 550, 20, 15, 10, 5);
        machine.makeCoffee(350, 75, 20, 7);
        assertEquals(400, machine.getWater());
        assertEquals(50, machine.getMilk());
        assertEquals(120, machine.getCoffeeBeans());
        assertEquals(9, machine.getCups());
        assertEquals(550, machine.getMoney());
    }

    @Test
    void testMakeCoffeeNotEnoughBeans() {
        CoffeeMachine machine = new CoffeeMachine(400, 540, 10, 9, 550, 20, 15, 10, 5);
        machine.makeCoffee(200, 100, 12, 6);
        assertEquals(400, machine.getWater());
        assertEquals(540, machine.getMilk());
        assertEquals(10, machine.getCoffeeBeans());
        assertEquals(9, machine.getCups());
        assertEquals(550, machine.getMoney());
    }

    @Test
    void testMakeCoffeeNotEnoughCups() {
        CoffeeMachine machine = new CoffeeMachine(400, 540, 120, 0, 550, 20, 15, 10, 5);
        machine.makeCoffee(200, 100, 12, 6);
        assertEquals(400, machine.getWater());
        assertEquals(540, machine.getMilk());
        assertEquals(120, machine.getCoffeeBeans());
        assertEquals(0, machine.getCups());
        assertEquals(550, machine.getMoney());
    }
    
    @Test
    void testSellSnackCookieSuccess() {
        CoffeeMachine machine = new CoffeeMachine(400, 540, 120, 9, 550, 20, 15, 10, 5);
        machine.sellSnack(1);
        assertEquals(19, machine.getCookies());
        assertEquals(552, machine.getMoney());
    }

    @Test
    void testSellSnackBrownieSuccess() {
        CoffeeMachine machine = new CoffeeMachine(400, 540, 120, 9, 550, 20, 15, 10, 5);
        machine.sellSnack(2);
        assertEquals(14, machine.getBrownies());
        assertEquals(553, machine.getMoney());
    }

    @Test
    void testSellSnackTartSuccess() {
        CoffeeMachine machine = new CoffeeMachine(400, 540, 120, 9, 550, 20, 15, 10, 5);
        machine.sellSnack(3);
        assertEquals(9, machine.getTarts());
        assertEquals(554, machine.getMoney());
    }

    @Test
    void testSellSnackMochiSuccess() {
        CoffeeMachine machine = new CoffeeMachine(400, 540, 120, 9, 550, 20, 15, 10, 5);
        machine.sellSnack(4);
        assertEquals(4, machine.getMochi());
        assertEquals(555, machine.getMoney());
    }

    @Test
    void testSellSnackCookieOutOfStock() {
        CoffeeMachine machine = new CoffeeMachine(400, 540, 120, 9, 550, 0, 15, 10, 5);
        machine.sellSnack(1);
        assertEquals(0, machine.getCookies());
        assertEquals(550, machine.getMoney());
    }

    @Test
    void testSellSnackBrownieOutOfStock() {
        CoffeeMachine machine = new CoffeeMachine(400, 540, 120, 9, 550, 20, 0, 10, 5);
        machine.sellSnack(2);
        assertEquals(0, machine.getBrownies());
        assertEquals(550, machine.getMoney());
    }

    @Test
    void testSellSnackTartOutOfStock() {
        CoffeeMachine machine = new CoffeeMachine(400, 540, 120, 9, 550, 20, 15, 0, 5);
        machine.sellSnack(3);
        assertEquals(0, machine.getTarts());
        assertEquals(550, machine.getMoney());
    }

    @Test
    void testSellSnackMochiOutOfStock() {
        CoffeeMachine machine = new CoffeeMachine(400, 540, 120, 9, 550, 20, 15, 10, 0);
        machine.sellSnack(4);
        assertEquals(0, machine.getMochi());
        assertEquals(550, machine.getMoney());
    }

    @Test
    void testSellSnackInvalidChoice() {
        CoffeeMachine machine = new CoffeeMachine(400, 540, 120, 9, 550, 20, 15, 10, 5);
        machine.sellSnack(5);
        assertEquals(20, machine.getCookies());
        assertEquals(15, machine.getBrownies());
        assertEquals(10, machine.getTarts());
        assertEquals(5, machine.getMochi());
        assertEquals(550, machine.getMoney());
    }
} 
