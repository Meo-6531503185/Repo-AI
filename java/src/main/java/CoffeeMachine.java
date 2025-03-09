import java.util.Scanner;

public class CoffeeMachine {

    private int water;
    private int milk;
    private int coffeeBeans;
    private int cups;
    private int money;
    private int cookies;
    private int brownies;
    private int tarts;
    private int mochi;

    public CoffeeMachine(int water, int milk, int coffeeBeans, int cups, int money, int cookies, int brownies, int tarts, int mochi) {
        this.water = water;
        this.milk = milk;
        this.coffeeBeans = coffeeBeans;
        this.cups = cups;
        this.money = money;
        this.cookies = cookies;
        this.brownies = brownies;
        this.tarts = tarts;
        this.mochi = mochi;
    }

    public int getWater() {
        return water;
    }

    public int getMilk() {
        return milk;
    }

    public int getCoffeeBeans() {
        return coffeeBeans;
    }

    public int getCups() {
        return cups;
    }

    public int getMoney() {
        return money;
    }

    public int getCookies() {
        return cookies;
    }

    public int getBrownies() {
        return brownies;
    }

    public int getTarts() {
        return tarts;
    }

    public int getMochi() {
        return mochi;
    }

    public void printStatus() {
        System.out.println("The coffee machine has:");
        System.out.println(water + " ml of water");
        System.out.println(milk + " ml of milk");
        System.out.println(coffeeBeans + " g of coffee beans");
        System.out.println(cups + " disposable cups");
        System.out.println("$" + money + " of money");
        System.out.println(cookies + " cookies");
        System.out.println(brownies + " brownies");
        System.out.println(tarts + " tarts");
        System.out.println(mochi + " mochi");
    }

    public void makeCoffee(int waterNeeded, int milkNeeded, int coffeeNeeded, int cost) {
        if (water < waterNeeded) {
            System.out.println("Sorry, not enough water!");
        } else if (milk < milkNeeded) {
            System.out.println("Sorry, not enough milk!");
        } else if (coffeeBeans < coffeeNeeded) {
            System.out.println("Sorry, not enough coffee beans!");
        } else if (cups < 1) {
            System.out.println("Sorry, not enough cups!");
        } else {
            System.out.println("I have enough resources, making you a coffee!");
            water -= waterNeeded;
            milk -= milkNeeded;
            coffeeBeans -= coffeeNeeded;
            cups--;
            money += cost;
        }
    }
    
    public void sellSnack(int choice) {
        switch (choice) {
            case 1:
                if (cookies > 0) {
                    System.out.println("Enjoy your cookie!");
                    cookies--;
                    money += 2; 
                } else {
                    System.out.println("Sorry, out of cookies!");
                }
                break;
            case 2:
                if (brownies > 0) {
                    System.out.println("Enjoy your brownie!");
                    brownies--;
                    money += 3; 
                } else {
                    System.out.println("Sorry, out of brownies!");
                }
                break;
            case 3:
                if (tarts > 0) {
                    System.out.println("Enjoy your tart!");
                    tarts--;
                    money += 4; 
                } else {
                    System.out.println("Sorry, out of tarts!");
                }
                break;
            case 4:
                if (mochi > 0) {
                    System.out.println("Enjoy your mochi!");
                    mochi--;
                    money += 5;
                } else {
                    System.out.println("Sorry, out of mochi!");
                }
                break;
            default:
                System.out.println("Invalid choice.");
        }
    }

    public static void main(String[] args) {
        CoffeeMachine machine = new CoffeeMachine(400, 540, 120, 9, 550, 20, 15, 10, 5); 
        Scanner scanner = new Scanner(System.in);

        while (true) {
            System.out.println("Write action (buy, fill, take, remaining, snack, exit):");
            String action = scanner.next();

            switch (action) {
                case "buy":
                    System.out.println("What do you want to buy? 1 - espresso, 2 - latte, 3 - cappuccino:");
                    int choice = scanner.nextInt();
                    if (choice == 1) {
                        machine.makeCoffee(250, 0, 16, 4);
                    } else if (choice == 2) {
                        machine.makeCoffee(350, 75, 20, 7);
                    } else if (choice == 3) {
                        machine.makeCoffee(200, 100, 12, 6);
                    }
                    break;
                case "fill":
                    System.out.println("Write how many ml of water you want to add:");
                    machine.water += scanner.nextInt();
                    System.out.println("Write how many ml of milk you want to add:");
                    machine.milk += scanner.nextInt();
                    System.out.println("Write how many grams of coffee beans you want to add:");
                    machine.coffeeBeans += scanner.nextInt();
                    System.out.println("Write how many disposable cups you want to add:");
                    machine.cups += scanner.nextInt();
                    break;
                case "take":
                    System.out.println("I gave you $" + machine.money);
                    machine.money = 0;
                    break;
                case "remaining":
                    machine.printStatus();
                    break;
                case "snack":
                    System.out.println("What snack do you want? 1 - Cookie ($2), 2 - Brownie ($3), 3 - Tart ($4), 4 - Mochi ($5):");
                    int snackChoice = scanner.nextInt();
                    machine.sellSnack(snackChoice);
                    break;
                case "exit":
                    System.out.println("Exiting... Goodbye!");
                    scanner.close();
                    return;
                default:
                    System.out.println("Unknown action");
                    break;
            }
        }
    }
}
