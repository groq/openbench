public class DaHeck {
    private static boolean check_flag(String s) {
        char[] cs = s.toCharArray();
        char[] daheck = new char[cs.length];
        int n = cs.length ^ daheck.length;
        char[] heck = "001002939948347799120432047441372907443274204020958757273".toCharArray();

        while (true) {

            try {
                if (heck[n] - cs[n % cs.length] < 0) daheck[n] = (char) (heck[n] - cs[n % cs.length] % 128);
                else daheck[n] = (char) (heck[n] - cs[n % cs.length] % 255);

                n++;
            } catch (Throwable t) {
                break;
            }
        }

        return "\uffc8\uffbd\uffce\uffbc\uffca\uffb7\uffc5\uffcb\u0005\uffc5\uffd5\uffc1\uffff\uffc1\uffd8\uffd1\uffc4\uffcb\u0010\uffd3\uffc4\u0001\uffbf\uffbf\uffd1\uffc0\uffc5\uffbb\uffd5\uffbe\u0003\uffca\uffff\uffda\uffc3\u0007\uffc2\u0001\uffd4\uffc0\u0004\uffbe\uffff\uffbe\uffc1\ufffd\uffb5".equals(new String(daheck));
    }

    public static void main(String... args) {
        if (args.length != 1) {
            System.out.println(":thonk:");
            System.exit(1);
        }

        if (args[0].length() != 47) {
            System.out.println(":thonkeng:");
            System.exit(1);
        }

        if (check_flag(args[0])) System.out.println("Huh. How'd you know?");
        else System.out.println("Da heck? No.");
    }
}
